// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
int sdpa_prefill_bf16s_avx512bf16(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt);
int sdpa_kvcache_bf16s_avx512bf16(const Mat& query, const Mat& past_key, const Mat& past_value, const Mat& cur_key, const Mat& cur_value, Mat& cached_key, Mat& cached_value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
int sdpa_prefill_bf16s_avx2(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt);
int sdpa_kvcache_bf16s_avx2(const Mat& query, const Mat& past_key, const Mat& past_value, const Mat& cur_key, const Mat& cur_value, Mat& cached_key, Mat& cached_value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt);
#endif

static void sdpa_pack_query_bf16s(const Mat& query_head, Mat& queryT, int i, int max_ii, size_t q_hstep)
{
    const int head_dim = query_head.w;
    unsigned short* pp = queryT;
    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const int i0 = i + ii;
        const unsigned short* qptr = (const unsigned short*)query_head + (size_t)i0 * q_hstep;

        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m256i _r0 = _mm256_loadu_si256((const __m256i*)qptr);
            __m256i _r1 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep));
            __m256i _r2 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 2));
            __m256i _r3 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 3));
            __m256i _r4 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 4));
            __m256i _r5 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 5));
            __m256i _r6 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 6));
            __m256i _r7 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 7));
            __m256i _r8 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 8));
            __m256i _r9 = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 9));
            __m256i _ra = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 10));
            __m256i _rb = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 11));
            __m256i _rc = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 12));
            __m256i _rd = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 13));
            __m256i _re = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 14));
            __m256i _rf = _mm256_loadu_si256((const __m256i*)(qptr + q_hstep * 15));

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

            _mm512_storeu_si512((__m512i*)pp, _p0);
            _mm512_storeu_si512((__m512i*)(pp + 32), _p1);
            _mm512_storeu_si512((__m512i*)(pp + 64), _p2);
            _mm512_storeu_si512((__m512i*)(pp + 96), _p3);
            _mm512_storeu_si512((__m512i*)(pp + 128), _p4);
            _mm512_storeu_si512((__m512i*)(pp + 160), _p5);
            _mm512_storeu_si512((__m512i*)(pp + 192), _p6);
            _mm512_storeu_si512((__m512i*)(pp + 224), _p7);
#else
            transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

            _mm256_storeu_si256((__m256i*)pp, _r0);
            _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
            _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
            _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
            _mm256_storeu_si256((__m256i*)(pp + 64), _r4);
            _mm256_storeu_si256((__m256i*)(pp + 80), _r5);
            _mm256_storeu_si256((__m256i*)(pp + 96), _r6);
            _mm256_storeu_si256((__m256i*)(pp + 112), _r7);
            _mm256_storeu_si256((__m256i*)(pp + 128), _r8);
            _mm256_storeu_si256((__m256i*)(pp + 144), _r9);
            _mm256_storeu_si256((__m256i*)(pp + 160), _ra);
            _mm256_storeu_si256((__m256i*)(pp + 176), _rb);
            _mm256_storeu_si256((__m256i*)(pp + 192), _rc);
            _mm256_storeu_si256((__m256i*)(pp + 208), _rd);
            _mm256_storeu_si256((__m256i*)(pp + 224), _re);
            _mm256_storeu_si256((__m256i*)(pp + 240), _rf);
#endif // __AVX512BF16__
            qptr += 16;
            pp += 256;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = qptr[0];
            pp[1] = qptr[1];
            pp[2] = qptr[q_hstep];
            pp[3] = qptr[q_hstep + 1];
            pp[4] = qptr[q_hstep * 2];
            pp[5] = qptr[q_hstep * 2 + 1];
            pp[6] = qptr[q_hstep * 3];
            pp[7] = qptr[q_hstep * 3 + 1];
            pp[8] = qptr[q_hstep * 4];
            pp[9] = qptr[q_hstep * 4 + 1];
            pp[10] = qptr[q_hstep * 5];
            pp[11] = qptr[q_hstep * 5 + 1];
            pp[12] = qptr[q_hstep * 6];
            pp[13] = qptr[q_hstep * 6 + 1];
            pp[14] = qptr[q_hstep * 7];
            pp[15] = qptr[q_hstep * 7 + 1];
            pp[16] = qptr[q_hstep * 8];
            pp[17] = qptr[q_hstep * 8 + 1];
            pp[18] = qptr[q_hstep * 9];
            pp[19] = qptr[q_hstep * 9 + 1];
            pp[20] = qptr[q_hstep * 10];
            pp[21] = qptr[q_hstep * 10 + 1];
            pp[22] = qptr[q_hstep * 11];
            pp[23] = qptr[q_hstep * 11 + 1];
            pp[24] = qptr[q_hstep * 12];
            pp[25] = qptr[q_hstep * 12 + 1];
            pp[26] = qptr[q_hstep * 13];
            pp[27] = qptr[q_hstep * 13 + 1];
            pp[28] = qptr[q_hstep * 14];
            pp[29] = qptr[q_hstep * 14 + 1];
            pp[30] = qptr[q_hstep * 15];
            pp[31] = qptr[q_hstep * 15 + 1];
            qptr += 2;
            pp += 32;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pp[0] = qptr[0];
            pp[1] = qptr[q_hstep];
            pp[2] = qptr[q_hstep * 2];
            pp[3] = qptr[q_hstep * 3];
            pp[4] = qptr[q_hstep * 4];
            pp[5] = qptr[q_hstep * 5];
            pp[6] = qptr[q_hstep * 6];
            pp[7] = qptr[q_hstep * 7];
            pp[8] = qptr[q_hstep * 8];
            pp[9] = qptr[q_hstep * 9];
            pp[10] = qptr[q_hstep * 10];
            pp[11] = qptr[q_hstep * 11];
            pp[12] = qptr[q_hstep * 12];
            pp[13] = qptr[q_hstep * 13];
            pp[14] = qptr[q_hstep * 14];
            pp[15] = qptr[q_hstep * 15];
            qptr++;
            pp += 16;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const int i0 = i + ii;
        const unsigned short* qptr = (const unsigned short*)query_head + (size_t)i0 * q_hstep;

        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)qptr);
            __m128i _r1 = _mm_loadu_si128((const __m128i*)(qptr + q_hstep));
            __m128i _r2 = _mm_loadu_si128((const __m128i*)(qptr + q_hstep * 2));
            __m128i _r3 = _mm_loadu_si128((const __m128i*)(qptr + q_hstep * 3));
            __m128i _r4 = _mm_loadu_si128((const __m128i*)(qptr + q_hstep * 4));
            __m128i _r5 = _mm_loadu_si128((const __m128i*)(qptr + q_hstep * 5));
            __m128i _r6 = _mm_loadu_si128((const __m128i*)(qptr + q_hstep * 6));
            __m128i _r7 = _mm_loadu_si128((const __m128i*)(qptr + q_hstep * 7));

#if __AVX512BF16__
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            transpose4x4_epi32(_r4, _r5, _r6, _r7);

            __m256i _p0 = combine4x2_epi32(_r0, _r4);
            __m256i _p1 = combine4x2_epi32(_r1, _r5);
            __m256i _p2 = combine4x2_epi32(_r2, _r6);
            __m256i _p3 = combine4x2_epi32(_r3, _r7);

            _mm256_storeu_si256((__m256i*)pp, _p0);
            _mm256_storeu_si256((__m256i*)(pp + 16), _p1);
            _mm256_storeu_si256((__m256i*)(pp + 32), _p2);
            _mm256_storeu_si256((__m256i*)(pp + 48), _p3);
#else
            transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

            _mm_storeu_si128((__m128i*)pp, _r0);
            _mm_storeu_si128((__m128i*)(pp + 8), _r1);
            _mm_storeu_si128((__m128i*)(pp + 16), _r2);
            _mm_storeu_si128((__m128i*)(pp + 24), _r3);
            _mm_storeu_si128((__m128i*)(pp + 32), _r4);
            _mm_storeu_si128((__m128i*)(pp + 40), _r5);
            _mm_storeu_si128((__m128i*)(pp + 48), _r6);
            _mm_storeu_si128((__m128i*)(pp + 56), _r7);
#endif // __AVX512BF16__
            qptr += 8;
            pp += 64;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = qptr[0];
            pp[1] = qptr[1];
            pp[2] = qptr[q_hstep];
            pp[3] = qptr[q_hstep + 1];
            pp[4] = qptr[q_hstep * 2];
            pp[5] = qptr[q_hstep * 2 + 1];
            pp[6] = qptr[q_hstep * 3];
            pp[7] = qptr[q_hstep * 3 + 1];
            pp[8] = qptr[q_hstep * 4];
            pp[9] = qptr[q_hstep * 4 + 1];
            pp[10] = qptr[q_hstep * 5];
            pp[11] = qptr[q_hstep * 5 + 1];
            pp[12] = qptr[q_hstep * 6];
            pp[13] = qptr[q_hstep * 6 + 1];
            pp[14] = qptr[q_hstep * 7];
            pp[15] = qptr[q_hstep * 7 + 1];
            qptr += 2;
            pp += 16;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pp[0] = qptr[0];
            pp[1] = qptr[q_hstep];
            pp[2] = qptr[q_hstep * 2];
            pp[3] = qptr[q_hstep * 3];
            pp[4] = qptr[q_hstep * 4];
            pp[5] = qptr[q_hstep * 5];
            pp[6] = qptr[q_hstep * 6];
            pp[7] = qptr[q_hstep * 7];
            qptr++;
            pp += 8;
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int i0 = i + ii;
        const unsigned short* qptr = (const unsigned short*)query_head + (size_t)i0 * q_hstep;

        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)qptr);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)(qptr + q_hstep));
            __m128i _r2 = _mm_loadl_epi64((const __m128i*)(qptr + q_hstep * 2));
            __m128i _r3 = _mm_loadl_epi64((const __m128i*)(qptr + q_hstep * 3));

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
            _mm_storeu_si128((__m128i*)pp, _r0);
            _mm_storeu_si128((__m128i*)(pp + 8), _r1);
            qptr += 4;
            pp += 16;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = qptr[0];
            pp[1] = qptr[1];
            pp[2] = qptr[q_hstep];
            pp[3] = qptr[q_hstep + 1];
            pp[4] = qptr[q_hstep * 2];
            pp[5] = qptr[q_hstep * 2 + 1];
            pp[6] = qptr[q_hstep * 3];
            pp[7] = qptr[q_hstep * 3 + 1];
            qptr += 2;
            pp += 8;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pp[0] = qptr[0];
            pp[1] = qptr[q_hstep];
            pp[2] = qptr[q_hstep * 2];
            pp[3] = qptr[q_hstep * 3];
            qptr++;
            pp += 4;
        }
    }
#endif // __SSE2__
    for (; ii < max_ii; ii++)
    {
        const unsigned short* qptr = (const unsigned short*)query_head + (size_t)(i + ii) * q_hstep;
        memcpy(pp, qptr, (size_t)head_dim * sizeof(unsigned short));
        pp += head_dim;
    }
}

static void sdpa_pack_mask_tile_bf16s(const Mat& mask_head, Mat& packed_mask_tile, int max_ii, size_t mask_hstep)
{
    const int key_seqlen = mask_head.w;
    const unsigned short* mask_ptr = mask_head;
    unsigned short* pp = packed_mask_tile;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const unsigned short* maskptr = mask_ptr + (size_t)ii * mask_hstep;
        for (int j = 0; j < key_seqlen; j++)
        {
            pp[0] = maskptr[0];
            pp[1] = maskptr[mask_hstep];
            pp[2] = maskptr[mask_hstep * 2];
            pp[3] = maskptr[mask_hstep * 3];
            pp[4] = maskptr[mask_hstep * 4];
            pp[5] = maskptr[mask_hstep * 5];
            pp[6] = maskptr[mask_hstep * 6];
            pp[7] = maskptr[mask_hstep * 7];
            pp[8] = maskptr[mask_hstep * 8];
            pp[9] = maskptr[mask_hstep * 9];
            pp[10] = maskptr[mask_hstep * 10];
            pp[11] = maskptr[mask_hstep * 11];
            pp[12] = maskptr[mask_hstep * 12];
            pp[13] = maskptr[mask_hstep * 13];
            pp[14] = maskptr[mask_hstep * 14];
            pp[15] = maskptr[mask_hstep * 15];
            maskptr++;
            pp += 16;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* maskptr = mask_ptr + (size_t)ii * mask_hstep;
        for (int j = 0; j < key_seqlen; j++)
        {
            pp[0] = maskptr[0];
            pp[1] = maskptr[mask_hstep];
            pp[2] = maskptr[mask_hstep * 2];
            pp[3] = maskptr[mask_hstep * 3];
            pp[4] = maskptr[mask_hstep * 4];
            pp[5] = maskptr[mask_hstep * 5];
            pp[6] = maskptr[mask_hstep * 6];
            pp[7] = maskptr[mask_hstep * 7];
            maskptr++;
            pp += 8;
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* maskptr = mask_ptr + (size_t)ii * mask_hstep;
        for (int j = 0; j < key_seqlen; j++)
        {
            pp[0] = maskptr[0];
            pp[1] = maskptr[mask_hstep];
            pp[2] = maskptr[mask_hstep * 2];
            pp[3] = maskptr[mask_hstep * 3];
            maskptr++;
            pp += 4;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* maskptr0 = mask_ptr + (size_t)ii * mask_hstep;
        const unsigned short* maskptr1 = maskptr0 + mask_hstep;
        memcpy(pp, maskptr0, (size_t)key_seqlen * sizeof(unsigned short));
        memcpy(pp + key_seqlen, maskptr1, (size_t)key_seqlen * sizeof(unsigned short));
        pp += key_seqlen * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* maskptr = mask_ptr + (size_t)ii * mask_hstep;
        memcpy(pp, maskptr, (size_t)key_seqlen * sizeof(unsigned short));
        pp += key_seqlen;
    }
}

// packed_mask[mask_head][query_block][query_panel][key][query_lane] in bf16
static void sdpa_pack_mask_bf16s(const Mat& attn_mask_blob, Mat& packed_mask, int block_m, const Option& opt)
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
        const Mat mask_tile = mask_head.row_range(i0, max_ii);
        sdpa_pack_mask_tile_bf16s(mask_tile, packed_mask_tile, max_ii, mask_head.w * mask_head.elempack);
    }
}

// packed_key[token_panel][head_dim][token_lane] in bf16
static void sdpa_pack_key_tile_bf16s(const Mat& key, Mat& packed_key, int src_begin, int dst_begin, int max_seqlen)
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
    unsigned short* panel = packed_key;
    int j = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
    for (; j + 15 < max_seqlen; j += 16)
    {
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);

#if __AVX512BF16__
        unsigned short* pp = panel + (token_lane + j) * 2;
#else
        unsigned short* pp = panel + token_lane + j;
#endif // __AVX512BF16__
        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
            __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + hstep));
            __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 2));
            __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 3));
            __m256i _r4 = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 4));
            __m256i _r5 = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 5));
            __m256i _r6 = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 6));
            __m256i _r7 = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 7));
            __m256i _r8 = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 8));
            __m256i _r9 = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 9));
            __m256i _ra = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 10));
            __m256i _rb = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 11));
            __m256i _rc = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 12));
            __m256i _rd = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 13));
            __m256i _re = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 14));
            __m256i _rf = _mm256_loadu_si256((const __m256i*)(p0 + hstep * 15));

#if __AVX512BF16__
            transpose8x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            transpose8x8_epi32(_r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

            _mm512_storeu_si512((__m512i*)pp, combine8x2_epi32(_r0, _r8));
            _mm512_storeu_si512((__m512i*)(pp + panel_width * 2), combine8x2_epi32(_r1, _r9));
            _mm512_storeu_si512((__m512i*)(pp + panel_width * 4), combine8x2_epi32(_r2, _ra));
            _mm512_storeu_si512((__m512i*)(pp + panel_width * 6), combine8x2_epi32(_r3, _rb));
            _mm512_storeu_si512((__m512i*)(pp + panel_width * 8), combine8x2_epi32(_r4, _rc));
            _mm512_storeu_si512((__m512i*)(pp + panel_width * 10), combine8x2_epi32(_r5, _rd));
            _mm512_storeu_si512((__m512i*)(pp + panel_width * 12), combine8x2_epi32(_r6, _re));
            _mm512_storeu_si512((__m512i*)(pp + panel_width * 14), combine8x2_epi32(_r7, _rf));
#else
            transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

            _mm256_storeu_si256((__m256i*)pp, _r0);
            _mm256_storeu_si256((__m256i*)(pp + panel_width), _r1);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 2), _r2);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 3), _r3);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 4), _r4);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 5), _r5);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 6), _r6);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 7), _r7);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 8), _r8);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 9), _r9);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 10), _ra);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 11), _rb);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 12), _rc);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 13), _rd);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 14), _re);
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 15), _rf);
#endif // __AVX512BF16__

            p0 += 16;
            pp += panel_width * 16;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[hstep];
            pp[3] = p0[hstep + 1];
            pp[4] = p0[hstep * 2];
            pp[5] = p0[hstep * 2 + 1];
            pp[6] = p0[hstep * 3];
            pp[7] = p0[hstep * 3 + 1];
            pp[8] = p0[hstep * 4];
            pp[9] = p0[hstep * 4 + 1];
            pp[10] = p0[hstep * 5];
            pp[11] = p0[hstep * 5 + 1];
            pp[12] = p0[hstep * 6];
            pp[13] = p0[hstep * 6 + 1];
            pp[14] = p0[hstep * 7];
            pp[15] = p0[hstep * 7 + 1];
            pp[16] = p0[hstep * 8];
            pp[17] = p0[hstep * 8 + 1];
            pp[18] = p0[hstep * 9];
            pp[19] = p0[hstep * 9 + 1];
            pp[20] = p0[hstep * 10];
            pp[21] = p0[hstep * 10 + 1];
            pp[22] = p0[hstep * 11];
            pp[23] = p0[hstep * 11 + 1];
            pp[24] = p0[hstep * 12];
            pp[25] = p0[hstep * 12 + 1];
            pp[26] = p0[hstep * 13];
            pp[27] = p0[hstep * 13 + 1];
            pp[28] = p0[hstep * 14];
            pp[29] = p0[hstep * 14 + 1];
            pp[30] = p0[hstep * 15];
            pp[31] = p0[hstep * 15 + 1];

            p0 += 2;
            pp += panel_width * 2;
        }
#endif // __AVX512BF16__
        pp = panel + (size_t)d * panel_width + token_lane + j;
        for (; d < head_dim; d++)
        {
            pp[0] = *p0;
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
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);

#if __AVX512BF16__
        unsigned short* pp = panel + (token_lane + j) * 2;
#else
        unsigned short* pp = panel + token_lane + j;
#endif // __AVX512BF16__
        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + hstep));
            __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + hstep * 2));
            __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + hstep * 3));
            __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + hstep * 4));
            __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + hstep * 5));
            __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + hstep * 6));
            __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + hstep * 7));
#if __AVX512BF16__
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            transpose4x4_epi32(_r4, _r5, _r6, _r7);

            _mm256_storeu_si256((__m256i*)pp, combine4x2_epi32(_r0, _r4));
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 2), combine4x2_epi32(_r1, _r5));
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 4), combine4x2_epi32(_r2, _r6));
            _mm256_storeu_si256((__m256i*)(pp + panel_width * 6), combine4x2_epi32(_r3, _r7));
#else
            transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

            _mm_storeu_si128((__m128i*)pp, _r0);
            _mm_storeu_si128((__m128i*)(pp + panel_width), _r1);
            _mm_storeu_si128((__m128i*)(pp + panel_width * 2), _r2);
            _mm_storeu_si128((__m128i*)(pp + panel_width * 3), _r3);
            _mm_storeu_si128((__m128i*)(pp + panel_width * 4), _r4);
            _mm_storeu_si128((__m128i*)(pp + panel_width * 5), _r5);
            _mm_storeu_si128((__m128i*)(pp + panel_width * 6), _r6);
            _mm_storeu_si128((__m128i*)(pp + panel_width * 7), _r7);
#endif // __AVX512BF16__

            p0 += 8;
            pp += panel_width * 8;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[hstep];
            pp[3] = p0[hstep + 1];
            pp[4] = p0[hstep * 2];
            pp[5] = p0[hstep * 2 + 1];
            pp[6] = p0[hstep * 3];
            pp[7] = p0[hstep * 3 + 1];
            pp[8] = p0[hstep * 4];
            pp[9] = p0[hstep * 4 + 1];
            pp[10] = p0[hstep * 5];
            pp[11] = p0[hstep * 5 + 1];
            pp[12] = p0[hstep * 6];
            pp[13] = p0[hstep * 6 + 1];
            pp[14] = p0[hstep * 7];
            pp[15] = p0[hstep * 7 + 1];

            p0 += 2;
            pp += panel_width * 2;
        }
#endif // __AVX512BF16__
        pp = panel + (size_t)d * panel_width + token_lane + j;
        for (; d < head_dim; d++)
        {
            pp[0] = *p0;
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
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);

#if __AVX512BF16__
        unsigned short* pp = panel + (token_lane + j) * 2;
#else
        unsigned short* pp = panel + token_lane + j;
#endif // __AVX512BF16__
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + hstep));
            __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + hstep * 2));
            __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + hstep * 3));
#if __AVX512BF16__
            __m128i _tmp0 = _mm_unpacklo_epi32(_r0, _r1);
            __m128i _tmp1 = _mm_unpacklo_epi32(_r2, _r3);
            _r0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
            _r1 = _mm_unpackhi_epi64(_tmp0, _tmp1);

            _mm_storeu_si128((__m128i*)pp, _r0);
            _mm_storeu_si128((__m128i*)(pp + panel_width * 2), _r1);
#else
            __m128i _tmp0 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _tmp1 = _mm_unpacklo_epi16(_r2, _r3);
            _r0 = _mm_unpacklo_epi32(_tmp0, _tmp1);
            _r1 = _mm_unpackhi_epi32(_tmp0, _tmp1);

            _mm_storel_epi64((__m128i*)pp, _r0);
            _mm_storel_epi64((__m128i*)(pp + panel_width), _mm_srli_si128(_r0, 8));
            _mm_storel_epi64((__m128i*)(pp + panel_width * 2), _r1);
            _mm_storel_epi64((__m128i*)(pp + panel_width * 3), _mm_srli_si128(_r1, 8));
#endif // __AVX512BF16__

            p0 += 4;
            pp += panel_width * 4;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[hstep];
            pp[3] = p0[hstep + 1];
            pp[4] = p0[hstep * 2];
            pp[5] = p0[hstep * 2 + 1];
            pp[6] = p0[hstep * 3];
            pp[7] = p0[hstep * 3 + 1];

            p0 += 2;
            pp += panel_width * 2;
        }
#endif // __AVX512BF16__
        pp = panel + (size_t)d * panel_width + token_lane + j;
        for (; d < head_dim; d++)
        {
            pp[0] = *p0;
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
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);
        const unsigned short* p1 = p0 + hstep;
        int d = 0;
        unsigned short* pp;
#if __AVX512BF16__
        pp = panel + (token_lane + j) * 2;
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            p0 += 2;
            p1 += 2;
            pp += panel_width * 2;
        }
#endif // __AVX512BF16__
        pp = panel + (size_t)d * panel_width + token_lane + j;
        for (; d < head_dim; d++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            p0++;
            p1++;
            pp += panel_width;
        }
    }
    for (; j < max_seqlen; j++)
    {
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);
        int d = 0;
        unsigned short* pp;
#if __AVX512BF16__
        pp = panel + (token_lane + j) * 2;
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            p0 += 2;
            pp += panel_width * 2;
        }
#endif // __AVX512BF16__
        pp = panel + (size_t)d * panel_width + token_lane + j;
        for (; d < head_dim; d++)
        {
            *pp = *p0++;
            pp += panel_width;
        }
    }
}

// packed_value[token_panel][value_panel][token_lane][value_lane] in bf16
static void sdpa_pack_value_tile_bf16s(const Mat& value, Mat& packed_value, int src_begin, int dst_begin, int max_seqlen)
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
    unsigned short* panel = packed_value;
    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        unsigned short* pp = panel + (size_t)d * panel_width + token_lane * 16;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
            p0 += value_dim;
            pp += 16;
        }
    }
#endif // __AVX512F__
    for (; d + 7 < value_dim; d += 8)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        unsigned short* pp = panel + (size_t)d * panel_width + token_lane * 8;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
            p0 += value_dim;
            pp += 8;
        }
    }
#endif // __AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        unsigned short* pp = panel + (size_t)d * panel_width + token_lane * 4;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
            p0 += value_dim;
            pp += 4;
        }
    }
#endif // __SSE2__
    for (; d + 1 < value_dim; d += 2)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        unsigned short* pp = panel + (size_t)d * panel_width + token_lane * 2;
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
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        unsigned short* pp = panel + (size_t)d * panel_width + token_lane;
        for (int n = 0; n < max_seqlen; n++)
        {
            *pp++ = *p0;
            p0 += value_dim;
        }
    }
}

static void sdpa_append_kvcache_token_bf16s(const Mat& key, const Mat& value, Mat& cached_key, Mat& cached_value, int token_index, int panel_width)
{
    const int panel_id = token_index / panel_width;
    const int token_lane = token_index % panel_width;

    for (int g = 0; g < key.c; g++)
    {
        const unsigned short* kptr = key.channel(g);
        unsigned short* kpanel = (unsigned short*)cached_key.channel(g) + (size_t)panel_id * key.w * panel_width;
        int d = 0;
        unsigned short* kpp;
#if __AVX512BF16__
        kpp = kpanel + token_lane * 2;
        for (; d + 1 < key.w; d += 2)
        {
            kpp[0] = kptr[0];
            kpp[1] = kptr[1];
            kptr += 2;
            kpp += panel_width * 2;
        }
#endif // __AVX512BF16__
        kpp = kpanel + (size_t)d * panel_width + token_lane;
        for (; d < key.w; d++)
        {
            *kpp = *kptr++;
            kpp += panel_width;
        }

        const unsigned short* vptr = value.channel(g);
        unsigned short* vpanel = (unsigned short*)cached_value.channel(g) + (size_t)panel_id * value.w * panel_width;
        int vd = 0;
        unsigned short* vpp;
        const unsigned short* pV;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        vpp = vpanel + token_lane * 16;
        pV = vptr;
        for (; vd + 15 < value.w; vd += 16)
        {
            _mm256_storeu_si256((__m256i*)vpp, _mm256_loadu_si256((const __m256i*)pV));
            pV += 16;
            vpp += panel_width * 16;
        }
#endif // __AVX512F__
        vpp = vpanel + (size_t)vd * panel_width + token_lane * 8;
        pV = vptr + vd;
        for (; vd + 7 < value.w; vd += 8)
        {
            _mm_storeu_si128((__m128i*)vpp, _mm_loadu_si128((const __m128i*)pV));
            pV += 8;
            vpp += panel_width * 8;
        }
#endif // __AVX__
        vpp = vpanel + (size_t)vd * panel_width + token_lane * 4;
        pV = vptr + vd;
        for (; vd + 3 < value.w; vd += 4)
        {
            _mm_storel_epi64((__m128i*)vpp, _mm_loadl_epi64((const __m128i*)pV));
            pV += 4;
            vpp += panel_width * 4;
        }
#endif // __SSE2__
        vpp = vpanel + (size_t)vd * panel_width + token_lane * 2;
        pV = vptr + vd;
        for (; vd + 1 < value.w; vd += 2)
        {
            vpp[0] = pV[0];
            vpp[1] = pV[1];
            pV += 2;
            vpp += panel_width * 2;
        }
        vpp = vpanel + (size_t)vd * panel_width + token_lane;
        pV = vptr + vd;
        for (; vd < value.w; vd++)
        {
            *vpp = *pV++;
            vpp += panel_width;
        }
    }
}

// packed_value_fp32[key_block][value_panel][token][value_lane]
static void sdpa_pack_value_tile_bf16s_fp32(const Mat& packed_value, Mat& packed_value_fp32, int src_begin, int dst_begin, int max_seqlen, int dst_seqlen)
{
    const int value_dim = packed_value.w;
#if __AVX512F__
    const int panel_width = 16;
#elif __AVX__
    const int panel_width = 8;
#elif __SSE2__
    const int panel_width = 4;
#else
    const int panel_width = 2;
#endif
    const unsigned short* packed_value_ptr = packed_value;
    float* packed_value_fp32_ptr = packed_value_fp32;

    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin * 16;
        for (int jj = 0; jj < max_seqlen;)
        {
            const int token = src_begin + jj;
            const int token_lane = token % panel_width;
            const int max_nn = std::min(max_seqlen - jj, panel_width - token_lane);
            const unsigned short* value_panel = packed_value_ptr + (size_t)(token / panel_width) * value_dim * panel_width;
            const unsigned short* p0 = value_panel + (size_t)d * panel_width + token_lane * 16;
            for (int k = 0; k < max_nn; k++)
            {
                _mm512_storeu_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
                p0 += 16;
                pp += 16;
            }
            jj += max_nn;
        }
    }
#endif // __AVX512F__
    for (; d + 7 < value_dim; d += 8)
    {
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin * 8;
        for (int jj = 0; jj < max_seqlen;)
        {
            const int token = src_begin + jj;
            const int token_lane = token % panel_width;
            const int max_nn = std::min(max_seqlen - jj, panel_width - token_lane);
            const unsigned short* value_panel = packed_value_ptr + (size_t)(token / panel_width) * value_dim * panel_width;
            const unsigned short* p0 = value_panel + (size_t)d * panel_width + token_lane * 8;
            for (int k = 0; k < max_nn; k++)
            {
                _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                p0 += 8;
                pp += 8;
            }
            jj += max_nn;
        }
    }
#endif // __AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin * 4;
        for (int jj = 0; jj < max_seqlen;)
        {
            const int token = src_begin + jj;
            const int token_lane = token % panel_width;
            const int max_nn = std::min(max_seqlen - jj, panel_width - token_lane);
            const unsigned short* value_panel = packed_value_ptr + (size_t)(token / panel_width) * value_dim * panel_width;
            const unsigned short* p0 = value_panel + (size_t)d * panel_width + token_lane * 4;
            for (int k = 0; k < max_nn; k++)
            {
                _mm_storeu_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                p0 += 4;
                pp += 4;
            }
            jj += max_nn;
        }
    }
#endif // __SSE2__
    for (; d + 1 < value_dim; d += 2)
    {
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin * 2;
        for (int jj = 0; jj < max_seqlen;)
        {
            const int token = src_begin + jj;
            const int token_lane = token % panel_width;
            const int max_nn = std::min(max_seqlen - jj, panel_width - token_lane);
            const unsigned short* value_panel = packed_value_ptr + (size_t)(token / panel_width) * value_dim * panel_width;
            const unsigned short* p0 = value_panel + (size_t)d * panel_width + token_lane * 2;
            for (int k = 0; k < max_nn; k++)
            {
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p0[1]);
                pp += 2;
                p0 += 2;
            }
            jj += max_nn;
        }
    }
    for (; d < value_dim; d++)
    {
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin;
        for (int jj = 0; jj < max_seqlen;)
        {
            const int token = src_begin + jj;
            const int token_lane = token % panel_width;
            const int max_nn = std::min(max_seqlen - jj, panel_width - token_lane);
            const unsigned short* value_panel = packed_value_ptr + (size_t)(token / panel_width) * value_dim * panel_width;
            const unsigned short* p0 = value_panel + (size_t)d * panel_width + token_lane;
            for (int k = 0; k < max_nn; k++)
                *pp++ = bfloat16_to_float32(*p0++);
            jj += max_nn;
        }
    }
}

static void sdpa_pack_value_tile_bf16s_to_fp32(const Mat& value, Mat& packed_value_fp32, int src_begin, int dst_begin, int max_seqlen, int dst_seqlen)
{
    const int value_dim = value.w;
    float* packed_value_fp32_ptr = packed_value_fp32;

    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin * 16;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm512_storeu_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
            p0 += value_dim;
            pp += 16;
        }
    }
#endif // __AVX512F__
    for (; d + 7 < value_dim; d += 8)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin * 8;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
            p0 += value_dim;
            pp += 8;
        }
    }
#endif // __AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin * 4;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm_storeu_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
            p0 += value_dim;
            pp += 4;
        }
    }
#endif // __SSE2__
    for (; d + 1 < value_dim; d += 2)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin * 2;
        for (int n = 0; n < max_seqlen; n++)
        {
            pp[0] = bfloat16_to_float32(p0[0]);
            pp[1] = bfloat16_to_float32(p0[1]);
            p0 += value_dim;
            pp += 2;
        }
    }
    for (; d < value_dim; d++)
    {
        const unsigned short* p0 = value.row<const unsigned short>(src_begin) + d;
        float* pp = packed_value_fp32_ptr + (size_t)d * dst_seqlen + dst_begin;
        for (int n = 0; n < max_seqlen; n++)
        {
            *pp++ = bfloat16_to_float32(*p0);
            p0 += value_dim;
        }
    }
}

static void sdpa_attention_tile_bf16s(const Mat& queryT, const Mat& key_head, const Mat& packed_key_head, const Mat& value_head, const Mat& packed_value_head, const Mat& computation_value_head, const Mat& mask, size_t mask_hstep, const Mat& packed_mask, Mat& scoreT, Mat& outT, Mat& lT, int max_ii, float scale)
{
    const int head_dim = packed_key_head.empty() ? key_head.w : packed_key_head.w;
    const int key_seqlen = packed_key_head.empty() ? key_head.h : packed_key_head.h;
    const int TILE_M = lT.w;
    const int TILE_N = scoreT.w / TILE_M;
    const int value_dim = outT.w / TILE_M;
#if __AVX512F__
    const int NR = 16;
#elif __AVX__
    const int NR = 8;
#elif __SSE2__
    const int NR = 4;
#else
    const int NR = 2;
#endif

    const unsigned short* queryT_ptr = queryT;
    float* scoreT_ptr = scoreT;
    float* outT_ptr = outT;
    float* lptr = lT;
    const unsigned short* packed_mask_data = packed_mask;

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
            const unsigned short* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            __m512 _block_max = _mm512_set1_ps(-FLT_MAX);

            // qk
            {
                if (!key_head.empty())
                {
                    const unsigned short* pK = key_head.row<const unsigned short>(n);
                    float* pS = scoreptr;
                    const unsigned short* maskptr = mask.empty() ? 0 : (const unsigned short*)mask + n;
                    const __m512 _scale = _mm512_set1_ps(scale);
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m512 _sum0 = _mm512_setzero_ps();
                        __m512 _sum1 = _mm512_setzero_ps();
                        __m512 _sum2 = _mm512_setzero_ps();
                        __m512 _sum3 = _mm512_setzero_ps();
                        const unsigned short* pA = pQ;
                        int d = 0;
#if __AVX512BF16__
                        for (; d + 7 < head_dim; d += 8)
                        {
                            _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_mm512_loadu_si512((const __m512i*)pA), (__m512bh)_mm512_set1_epi32(((const int*)pK)[0]));
                            _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_mm512_loadu_si512((const __m512i*)(pA + 32)), (__m512bh)_mm512_set1_epi32(((const int*)(pK + 2))[0]));
                            _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_mm512_loadu_si512((const __m512i*)(pA + 64)), (__m512bh)_mm512_set1_epi32(((const int*)(pK + 4))[0]));
                            _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_mm512_loadu_si512((const __m512i*)(pA + 96)), (__m512bh)_mm512_set1_epi32(((const int*)(pK + 6))[0]));
                            pA += 128;
                            pK += 8;
                        }
                        for (; d + 1 < head_dim; d += 2)
                        {
                            _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_mm512_loadu_si512((const __m512i*)pA), (__m512bh)_mm512_set1_epi32(((const int*)pK)[0]));
                            pA += 32;
                            pK += 2;
                        }
#endif // __AVX512BF16__
                        for (; d + 3 < head_dim; d += 4)
                        {
                            _sum0 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA)), _mm512_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                            _sum1 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pA + 16))), _mm512_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                            _sum2 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pA + 32))), _mm512_set1_ps(bfloat16_to_float32(pK[2])), _sum2);
                            _sum3 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pA + 48))), _mm512_set1_ps(bfloat16_to_float32(pK[3])), _sum3);
                            pA += 64;
                            pK += 4;
                        }
                        for (; d < head_dim; d++)
                        {
                            _sum0 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA)), _mm512_set1_ps(bfloat16_to_float32(*pK)), _sum0);
                            pA += 16;
                            pK++;
                        }
                        __m512 _sum = _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3)), _scale);
                        if (maskptr)
                        {
                            if (mask_hstep == 0)
                            {
                                _sum = _mm512_add_ps(_sum, _mm512_set1_ps(bfloat16_to_float32(*maskptr)));
                            }
                            else
                            {
                                const __m256i _mask = _mm256_set_epi16(maskptr[mask_hstep * 15], maskptr[mask_hstep * 14], maskptr[mask_hstep * 13], maskptr[mask_hstep * 12], maskptr[mask_hstep * 11], maskptr[mask_hstep * 10], maskptr[mask_hstep * 9], maskptr[mask_hstep * 8], maskptr[mask_hstep * 7], maskptr[mask_hstep * 6], maskptr[mask_hstep * 5], maskptr[mask_hstep * 4], maskptr[mask_hstep * 3], maskptr[mask_hstep * 2], maskptr[mask_hstep], maskptr[0]);
                                _sum = _mm512_add_ps(_sum, bfloat2float_avx512(_mask));
                            }
                            maskptr++;
                        }
                        _mm512_storeu_ps(pS, _sum);
                        pS += 16;
                        _block_max = _mm512_max_ps(_block_max, _sum);
                    }
                }
                else
                {
                    const unsigned short* packed_maskptr = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + (size_t)n * 16 : 0;
                    const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)n * head_dim;
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

                            const unsigned short* pA = pQ;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)packed_maskptr)));
                                _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 16))));
                                _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 32))));
                                _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 48))));
                                _sum4 = _mm512_add_ps(_sum4, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 64))));
                                _sum5 = _mm512_add_ps(_sum5, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 80))));
                                _sum6 = _mm512_add_ps(_sum6, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 96))));
                                _sum7 = _mm512_add_ps(_sum7, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 112))));
                                _sum8 = _mm512_add_ps(_sum8, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 128))));
                                _sum9 = _mm512_add_ps(_sum9, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 144))));
                                _suma = _mm512_add_ps(_suma, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 160))));
                                _sumb = _mm512_add_ps(_sumb, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 176))));
                                _sumc = _mm512_add_ps(_sumc, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 192))));
                                _sumd = _mm512_add_ps(_sumd, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 208))));
                                _sume = _mm512_add_ps(_sume, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 224))));
                                _sumf = _mm512_add_ps(_sumf, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 240))));
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

                            const unsigned short* pA = pQ;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)packed_maskptr)));
                                _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 16))));
                                _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 32))));
                                _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 48))));
                                _sum4 = _mm512_add_ps(_sum4, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 64))));
                                _sum5 = _mm512_add_ps(_sum5, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 80))));
                                _sum6 = _mm512_add_ps(_sum6, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 96))));
                                _sum7 = _mm512_add_ps(_sum7, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 112))));
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
                            _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(_sum0, _sum1), _mm512_max_ps(_sum2, _sum3)), _mm512_max_ps(_mm512_max_ps(_sum4, _sum5), _mm512_max_ps(_sum6, _sum7))));
                        }
#endif // defined(__x86_64__) || defined(_M_X64)
                        for (; j + 3 < max_nn; j += 4)
                        {
                            __m512 _sum0 = _mm512_setzero_ps();
                            __m512 _sum1 = _mm512_setzero_ps();
                            __m512 _sum2 = _mm512_setzero_ps();
                            __m512 _sum3 = _mm512_setzero_ps();
                            const unsigned short* pA = pQ;
                            const unsigned short* pK = key_panel + j;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)packed_maskptr)));
                                _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 16))));
                                _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 32))));
                                _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 48))));
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
                            const unsigned short* pA = pQ;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)packed_maskptr)));
                                _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(packed_maskptr + 16))));
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
                            const unsigned short* pA = pQ;
                            int d = 0;
#if __AVX512BF16__
                            const int* pK_pair = (const int*)key_panel + j;
                            for (; d + 1 < head_dim; d += 2)
                            {
                                _sum = _mm512_dpbf16_ps(_sum, (__m512bh)_mm512_loadu_si512((const __m512i*)pA), (__m512bh)_mm512_set1_epi32(*pK_pair));
                                pA += 32;
                                pK_pair += 16;
                            }
#endif // __AVX512BF16__
                            const unsigned short* pK = key_panel + (size_t)d * NR + j;
                            for (; d < head_dim; d++)
                            {
                                const unsigned short k = *pK;
                                pK += 16;
                                _sum = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA)), _mm512_set1_ps(bfloat16_to_float32(k)), _sum);
                                pA += 16;
                            }
                            _sum = _mm512_mul_ps(_sum, _mm512_set1_ps(scale));
                            if (packed_maskptr)
                            {
                                _sum = _mm512_add_ps(_sum, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)packed_maskptr)));
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
                    const unsigned short* valueptr = value_head.row<const unsigned short>(n);
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
                        const unsigned short* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m512 _p = _mm512_loadu_ps(pS);
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
                        const unsigned short* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m512 _p = _mm512_loadu_ps(pS);
                            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[3])), _out3);
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
                        const unsigned short* pV = valueptr;
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
                else if (!computation_value_head.empty())
                {
                    const float* value_panel = computation_value_head.row(n / TILE_N);
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);

#if defined(__x86_64__) || defined(_M_X64)
                        if (value_panel_width == 16)
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
                            const float* pV = value_panel;
                            const float* pS = scoreptr;
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
                        }
                        else
#endif // defined(__x86_64__) || defined(_M_X64)
                        {
                            int lane = 0;
#if defined(__x86_64__) || defined(_M_X64)
                            for (; lane + 7 < value_panel_width; lane += 8)
                            {
                                float* outptr0 = outptr + (size_t)lane * 16;
                                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr0), _alpha);
                                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 16), _alpha);
                                __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 32), _alpha);
                                __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 48), _alpha);
                                __m512 _out4 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 64), _alpha);
                                __m512 _out5 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 80), _alpha);
                                __m512 _out6 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 96), _alpha);
                                __m512 _out7 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 112), _alpha);
                                const float* pV = value_panel + lane;
                                const float* pS = scoreptr;
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
                            }
#endif // defined(__x86_64__) || defined(_M_X64)
                            for (; lane + 3 < value_panel_width; lane += 4)
                            {
                                float* outptr0 = outptr + (size_t)lane * 16;
                                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr0), _alpha);
                                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 16), _alpha);
                                __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 32), _alpha);
                                __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 48), _alpha);
                                const float* pV = value_panel + lane;
                                const float* pS = scoreptr;
                                for (int j = 0; j < max_jj; j++)
                                {
                                    __m512 _p = _mm512_loadu_ps(pS);
                                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                                    _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                                    _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
                                    pS += 16;
                                    pV += value_panel_width;
                                }
                                _mm512_storeu_ps(outptr0, _out0);
                                _mm512_storeu_ps(outptr0 + 16, _out1);
                                _mm512_storeu_ps(outptr0 + 32, _out2);
                                _mm512_storeu_ps(outptr0 + 48, _out3);
                            }
                            for (; lane + 1 < value_panel_width; lane += 2)
                            {
                                float* outptr0 = outptr + (size_t)lane * 16;
                                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr0), _alpha);
                                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr0 + 16), _alpha);
                                const float* pV = value_panel + lane;
                                const float* pS = scoreptr;
                                for (int j = 0; j < max_jj; j++)
                                {
                                    __m512 _p = _mm512_loadu_ps(pS);
                                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                                    pS += 16;
                                    pV += value_panel_width;
                                }
                                _mm512_storeu_ps(outptr0, _out0);
                                _mm512_storeu_ps(outptr0 + 16, _out1);
                            }
                            for (; lane < value_panel_width; lane++)
                            {
                                __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr + (size_t)lane * 16), _alpha);
                                const float* pS = scoreptr;
                                const float* pV = value_panel + lane;
                                for (int j = 0; j < max_jj; j++)
                                {
                                    _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(*pV), _out);
                                    pS += 16;
                                    pV += value_panel_width;
                                }
                                _mm512_storeu_ps(outptr + (size_t)lane * 16, _out);
                            }
                        }
                        value_panel += (size_t)max_jj * value_panel_width;
                        outptr += value_panel_width * 16;
                        d += value_panel_width;
                    }
                }
                else
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
                            const float* pS = scoreT_ptr + (size_t)ii * TILE_N;
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV = value_panel;
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
                            const float* pS = scoreT_ptr + (size_t)ii * TILE_N;
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV = value_panel;
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 16;
                                const unsigned short* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m512 _p = _mm512_loadu_ps(pS);
                                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                                    _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                                    _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[3])), _out3);
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 16;
                                const unsigned short* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m512 _p = _mm512_loadu_ps(pS);
                                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[1])), _out1);
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 16;
                                const unsigned short* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(bfloat16_to_float32(*pV)), _out);
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
            const unsigned short* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);

            // qk
            {
                if (!key_head.empty())
                {
                    const unsigned short* pK = key_head.row<const unsigned short>(n);
                    float* pS = scoreptr;
                    const unsigned short* maskptr = mask.empty() ? 0 : (const unsigned short*)mask + n;
                    const __m256 _scale = _mm256_set1_ps(scale);
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m256 _sum0 = _mm256_setzero_ps();
                        __m256 _sum1 = _mm256_setzero_ps();
                        __m256 _sum2 = _mm256_setzero_ps();
                        __m256 _sum3 = _mm256_setzero_ps();
                        const unsigned short* pA = pQ;
                        int d = 0;
#if __AVX512BF16__
                        for (; d + 7 < head_dim; d += 8)
                        {
                            _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_mm256_loadu_si256((const __m256i*)pA), (__m256bh)_mm256_set1_epi32(((const int*)pK)[0]));
                            _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_mm256_loadu_si256((const __m256i*)(pA + 16)), (__m256bh)_mm256_set1_epi32(((const int*)(pK + 2))[0]));
                            _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_mm256_loadu_si256((const __m256i*)(pA + 32)), (__m256bh)_mm256_set1_epi32(((const int*)(pK + 4))[0]));
                            _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_mm256_loadu_si256((const __m256i*)(pA + 48)), (__m256bh)_mm256_set1_epi32(((const int*)(pK + 6))[0]));
                            pA += 64;
                            pK += 8;
                        }
                        for (; d + 1 < head_dim; d += 2)
                        {
                            _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_mm256_loadu_si256((const __m256i*)pA), (__m256bh)_mm256_set1_epi32(((const int*)pK)[0]));
                            pA += 16;
                            pK += 2;
                        }
#endif // __AVX512BF16__
                        for (; d + 3 < head_dim; d += 4)
                        {
                            _sum0 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA)), _mm256_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pA + 8))), _mm256_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pA + 16))), _mm256_set1_ps(bfloat16_to_float32(pK[2])), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pA + 24))), _mm256_set1_ps(bfloat16_to_float32(pK[3])), _sum3);
                            pA += 32;
                            pK += 4;
                        }
                        for (; d < head_dim; d++)
                        {
                            _sum0 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA)), _mm256_set1_ps(bfloat16_to_float32(*pK)), _sum0);
                            pA += 8;
                            pK++;
                        }
                        __m256 _sum = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3)), _scale);
                        if (maskptr)
                        {
                            if (mask_hstep == 0)
                            {
                                _sum = _mm256_add_ps(_sum, _mm256_set1_ps(bfloat16_to_float32(*maskptr)));
                            }
                            else
                            {
                                const __m128i _mask = _mm_set_epi16(maskptr[mask_hstep * 7], maskptr[mask_hstep * 6], maskptr[mask_hstep * 5], maskptr[mask_hstep * 4], maskptr[mask_hstep * 3], maskptr[mask_hstep * 2], maskptr[mask_hstep], maskptr[0]);
                                _sum = _mm256_add_ps(_sum, bfloat2float_avx(_mask));
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
                    const unsigned short* packed_maskptr = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + (size_t)n * 8 : 0;
                    const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)n * head_dim;
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
                            const unsigned short* pA = pQ;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)packed_maskptr)));
                                _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 8))));
                                _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 16))));
                                _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 24))));
                                _sum4 = _mm256_add_ps(_sum4, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 32))));
                                _sum5 = _mm256_add_ps(_sum5, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 40))));
                                _sum6 = _mm256_add_ps(_sum6, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 48))));
                                _sum7 = _mm256_add_ps(_sum7, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 56))));
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
                            const unsigned short* pA = pQ;
                            const unsigned short* pK = key_panel + j;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)packed_maskptr)));
                                _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 8))));
                                _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 16))));
                                _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 24))));
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
                            const unsigned short* pA = pQ;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)packed_maskptr)));
                                _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(packed_maskptr + 8))));
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
                            const unsigned short* pA = pQ;
                            const unsigned short* pK = key_panel + j;
                            int d = 0;
#if __AVX512BF16__
                            const int* pK_pair = (const int*)key_panel + j;
                            for (; d + 1 < head_dim; d += 2)
                            {
                                _sum = _mm256_dpbf16_ps(_sum, (__m256bh)_mm256_loadu_si256((const __m256i*)pA), (__m256bh)_mm256_set1_epi32(*pK_pair));
                                pA += 16;
                                pK_pair += NR;
                            }
                            pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                            for (; d < head_dim; d++)
                            {
                                _sum = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA)), _mm256_set1_ps(bfloat16_to_float32(*pK)), _sum);
                                pA += 8;
                                pK += NR;
                            }
                            _sum = _mm256_mul_ps(_sum, _mm256_set1_ps(scale));
                            if (packed_maskptr)
                            {
                                _sum = _mm256_add_ps(_sum, bfloat2float_avx(_mm_loadu_si128((const __m128i*)packed_maskptr)));
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
                _alpha = _mm256_and_ps(exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new))), _alpha_active);
                _m = _m_new;
                _l = _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _sum);
            }

            // pv
            {
                if (!value_head.empty())
                {
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    const unsigned short* valueptr = value_head.row<const unsigned short>(n);
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
                        const unsigned short* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m256 _p = _mm256_loadu_ps(pS);
                            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[3])), _out3);
                            _out4 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[4])), _out4);
                            _out5 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[5])), _out5);
                            _out6 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[6])), _out6);
                            _out7 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[7])), _out7);
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
                        const unsigned short* pV = valueptr;
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
                        const float* pS = scoreptr;
                        const unsigned short* pV = valueptr;
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
                else if (!computation_value_head.empty())
                {
                    const float* value_panel = computation_value_head.row(n / TILE_N);
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);

                        int lane = 0;
#if defined(__x86_64__) || defined(_M_X64)
                        for (; lane + 7 < value_panel_width; lane += 8)
                        {
                            float* outptr0 = outptr + (size_t)lane * 8;
                            __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr0), _alpha);
                            __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 8), _alpha);
                            __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 16), _alpha);
                            __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 24), _alpha);
                            __m256 _out4 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 32), _alpha);
                            __m256 _out5 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 40), _alpha);
                            __m256 _out6 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 48), _alpha);
                            __m256 _out7 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 56), _alpha);
                            const float* pS = scoreptr;
                            const float* pV = value_panel + lane;
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
#endif // defined(__x86_64__) || defined(_M_X64)
                        for (; lane + 3 < value_panel_width; lane += 4)
                        {
                            float* outptr0 = outptr + (size_t)lane * 8;
                            __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr0), _alpha);
                            __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 8), _alpha);
                            __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 16), _alpha);
                            __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 24), _alpha);
                            const float* pS = scoreptr;
                            const float* pV = value_panel + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m256 _p = _mm256_loadu_ps(pS);
                                _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                                _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                                _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[2]), _out2);
                                _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[3]), _out3);
                                pS += 8;
                                pV += value_panel_width;
                            }
                            _mm256_storeu_ps(outptr0, _out0);
                            _mm256_storeu_ps(outptr0 + 8, _out1);
                            _mm256_storeu_ps(outptr0 + 16, _out2);
                            _mm256_storeu_ps(outptr0 + 24, _out3);
                        }
                        for (; lane + 1 < value_panel_width; lane += 2)
                        {
                            float* outptr0 = outptr + (size_t)lane * 8;
                            __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr0), _alpha);
                            __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr0 + 8), _alpha);
                            const float* pS = scoreptr;
                            const float* pV = value_panel + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m256 _p = _mm256_loadu_ps(pS);
                                _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                                _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                                pS += 8;
                                pV += value_panel_width;
                            }
                            _mm256_storeu_ps(outptr0, _out0);
                            _mm256_storeu_ps(outptr0 + 8, _out1);
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr + (size_t)lane * 8), _alpha);
                            const float* pS = scoreptr;
                            const float* pV = value_panel + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(*pV), _out);
                                pS += 8;
                                pV += value_panel_width;
                            }
                            _mm256_storeu_ps(outptr + (size_t)lane * 8, _out);
                        }
                        value_panel += (size_t)max_jj * value_panel_width;
                        outptr += value_panel_width * 8;
                        d += value_panel_width;
                    }
                }
                else
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
                            const float* pS = scoreT_ptr + (size_t)ii * TILE_N;
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV = value_panel;
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 8;
                                const unsigned short* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m256 _p = _mm256_loadu_ps(pS);
                                    _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                                    _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                                    _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                                    _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[3])), _out3);
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 8;
                                const unsigned short* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m256 _p = _mm256_loadu_ps(pS);
                                    _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                                    _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[1])), _out1);
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 8;
                                const unsigned short* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(bfloat16_to_float32(*pV)), _out);
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
            const unsigned short* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);

            // qk
            {
                if (!key_head.empty())
                {
                    const unsigned short* pK = key_head.row<const unsigned short>(n);
                    float* pS = scoreptr;
                    const unsigned short* maskptr = mask.empty() ? 0 : (const unsigned short*)mask + n;
                    const __m128 _scale = _mm_set1_ps(scale);
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m128 _sum0 = _mm_setzero_ps();
                        __m128 _sum1 = _mm_setzero_ps();
                        __m128 _sum2 = _mm_setzero_ps();
                        __m128 _sum3 = _mm_setzero_ps();
                        const unsigned short* pA = pQ;
                        int d = 0;
#if __AVX512BF16__
                        for (; d + 7 < head_dim; d += 8)
                        {
                            _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_mm_loadu_si128((const __m128i*)pA), (__m128bh)_mm_set1_epi32(((const int*)pK)[0]));
                            _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_mm_loadu_si128((const __m128i*)(pA + 8)), (__m128bh)_mm_set1_epi32(((const int*)(pK + 2))[0]));
                            _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_mm_loadu_si128((const __m128i*)(pA + 16)), (__m128bh)_mm_set1_epi32(((const int*)(pK + 4))[0]));
                            _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_mm_loadu_si128((const __m128i*)(pA + 24)), (__m128bh)_mm_set1_epi32(((const int*)(pK + 6))[0]));
                            pA += 32;
                            pK += 8;
                        }
                        for (; d + 1 < head_dim; d += 2)
                        {
                            _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_mm_loadu_si128((const __m128i*)pA), (__m128bh)_mm_set1_epi32(((const int*)pK)[0]));
                            pA += 8;
                            pK += 2;
                        }
#endif // __AVX512BF16__
                        for (; d + 3 < head_dim; d += 4)
                        {
                            _sum0 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA)), _mm_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pA + 4))), _mm_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pA + 8))), _mm_set1_ps(bfloat16_to_float32(pK[2])), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pA + 12))), _mm_set1_ps(bfloat16_to_float32(pK[3])), _sum3);
                            pA += 16;
                            pK += 4;
                        }
                        for (; d < head_dim; d++)
                        {
                            _sum0 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA)), _mm_set1_ps(bfloat16_to_float32(*pK)), _sum0);
                            pA += 4;
                            pK++;
                        }
                        __m128 _sum = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_sum0, _sum1), _mm_add_ps(_sum2, _sum3)), _scale);
                        if (maskptr)
                        {
                            if (mask_hstep == 0)
                            {
                                _sum = _mm_add_ps(_sum, _mm_set1_ps(bfloat16_to_float32(*maskptr)));
                            }
                            else
                            {
                                const __m128i _mask = _mm_set_epi16(0, 0, 0, 0, maskptr[mask_hstep * 3], maskptr[mask_hstep * 2], maskptr[mask_hstep], maskptr[0]);
                                _sum = _mm_add_ps(_sum, bfloat2float_sse(_mask));
                            }
                            maskptr++;
                        }
                        _mm_storeu_ps(pS, _sum);
                        pS += 4;
                        _block_max = _mm_max_ps(_block_max, _sum);
                    }
                }
                else
                {
                    const unsigned short* packed_maskptr = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + (size_t)n * 4 : 0;
                    const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)n * head_dim;
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
                            const unsigned short* pA = pQ;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)packed_maskptr)));
                                _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(packed_maskptr + 4))));
                                _sum2 = _mm_add_ps(_sum2, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(packed_maskptr + 8))));
                                _sum3 = _mm_add_ps(_sum3, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(packed_maskptr + 12))));
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
                            const unsigned short* pA = pQ;
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)packed_maskptr)));
                                _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(packed_maskptr + 4))));
                                packed_maskptr += 8;
                            }
                            _mm_storeu_ps(score_panel, _sum0);
                            _mm_storeu_ps(score_panel + 4, _sum1);
                            score_panel += 8;
                            _block_max = _mm_max_ps(_block_max, _mm_max_ps(_sum0, _sum1));
                        }
                        for (; j < max_nn; j++)
                        {
                            __m128 _sum = _mm_setzero_ps();
                            const unsigned short* pK = key_panel + j;
                            const unsigned short* pA = pQ;
                            int d = 0;
#if __AVX512BF16__
                            const int* pK_pair = (const int*)key_panel + j;
                            for (; d + 1 < head_dim; d += 2)
                            {
                                _sum = _mm_dpbf16_ps(_sum, (__m128bh)_mm_loadu_si128((const __m128i*)pA), (__m128bh)_mm_set1_epi32(*pK_pair));
                                pA += 8;
                                pK_pair += NR;
                            }
                            pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                            for (; d < head_dim; d++)
                            {
                                _sum = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA)), _mm_set1_ps(bfloat16_to_float32(*pK)), _sum);
                                pA += 4;
                                pK += NR;
                            }
                            _sum = _mm_mul_ps(_sum, _mm_set1_ps(scale));
                            if (packed_maskptr)
                            {
                                _sum = _mm_add_ps(_sum, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)packed_maskptr)));
                                packed_maskptr += 4;
                            }
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
                    const unsigned short* valueptr = value_head.row<const unsigned short>(n);
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
                        const unsigned short* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m128 _p = _mm_loadu_ps(pS);
                            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[3])), _out3);
                            _out4 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[4])), _out4);
                            _out5 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[5])), _out5);
                            _out6 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[6])), _out6);
                            _out7 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[7])), _out7);
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
                        const unsigned short* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m128 _p = _mm_loadu_ps(pS);
                            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[3])), _out3);
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
                        const unsigned short* pV = valueptr;
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
                else if (!computation_value_head.empty())
                {
                    const float* value_panel = computation_value_head.row(n / TILE_N);
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
                            const float* pV = value_panel + lane;
                            for (int j = 0; j < max_jj; j++)
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
                        for (; lane + 1 < value_panel_width; lane += 2)
                        {
                            __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                            __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                            const float* pS = scoreptr;
                            const float* pV = value_panel + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m128 _p = _mm_loadu_ps(pS);
                                _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[0]), _out0);
                                _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[1]), _out1);
                                pS += 4;
                                pV += value_panel_width;
                            }
                            _mm_storeu_ps(outptr, _out0);
                            _mm_storeu_ps(outptr + 4, _out1);
                            outptr += 8;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                            const float* pS = scoreptr;
                            const float* pV = value_panel + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(*pV), _out);
                                pS += 4;
                                pV += value_panel_width;
                            }
                            _mm_storeu_ps(outptr, _out);
                            outptr += 4;
                        }
                        value_panel += (size_t)max_jj * value_panel_width;
                        d += value_panel_width;
                    }
                }
                else
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 4;
                                const unsigned short* pV = value_panel;
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 4;
                                const unsigned short* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m128 _p = _mm_loadu_ps(pS);
                                    _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                                    _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[1])), _out1);
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
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 4;
                                const unsigned short* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(bfloat16_to_float32(*pV)), _out);
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
            const unsigned short* pQ0 = queryT_ptr + (size_t)ii * head_dim;
            const unsigned short* pQ1 = pQ0 + head_dim;
            float* score0 = scoreT_ptr + (size_t)ii * TILE_N;
            float* score1 = score0 + TILE_N;
            const unsigned short* packed_mask0 = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + n : 0;
            const unsigned short* packed_mask1 = packed_mask0 ? packed_mask0 + key_seqlen : 0;
            float block_max0 = -FLT_MAX;
            float block_max1 = -FLT_MAX;

            // qk
            {
                const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)n * head_dim;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    float* scoreptr0 = score0 + jj;
                    float* scoreptr1 = score1 + jj;
                    const unsigned short* packed_maskptr0 = packed_mask0 ? packed_mask0 + jj : 0;
                    const unsigned short* packed_maskptr1 = packed_mask1 ? packed_mask1 + jj : 0;
                    int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; j + 15 < max_nn; j += 16)
                    {
                        __m512 _sum0 = _mm512_setzero_ps();
                        __m512 _sum1 = _mm512_setzero_ps();
                        const unsigned short* pA0 = pQ0;
                        const unsigned short* pA1 = pQ1;
                        const unsigned short* pK = key_panel + j;
                        int d = 0;
#if __AVX512BF16__
                        pK = key_panel + j * 2;
                        for (; d + 1 < head_dim; d += 2)
                        {
                            __m512i _k = _mm512_loadu_si512((const __m512i*)pK);
                            _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_mm512_set1_epi32(((const int*)pA0)[0]), (__m512bh)_k);
                            _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_mm512_set1_epi32(((const int*)pA1)[0]), (__m512bh)_k);
                            pA0 += 2;
                            pA1 += 2;
                            pK += NR * 2;
                        }
                        pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            __m512 _k = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pK));
                            _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(bfloat16_to_float32(*pA0++)), _k, _sum0);
                            _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(bfloat16_to_float32(*pA1++)), _k, _sum1);
                            pK += NR;
                        }
                        __m512 _scale = _mm512_set1_ps(scale);
                        _sum0 = _mm512_mul_ps(_sum0, _scale);
                        _sum1 = _mm512_mul_ps(_sum1, _scale);
                        if (packed_maskptr0)
                        {
                            _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)packed_maskptr0)));
                            _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)packed_maskptr1)));
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
                        const unsigned short* pA0 = pQ0;
                        const unsigned short* pA1 = pQ1;
                        const unsigned short* pK = key_panel + j;
                        int d = 0;
#if __AVX512BF16__
#if _MSC_VER
                        __m256 _sum2 = _mm256_setzero_ps();
                        __m256 _sum3 = _mm256_setzero_ps();
                        __m256i _mask = _mm256_set1_epi32(0xffff0000);
#endif
                        pK = key_panel + j * 2;
                        for (; d + 1 < head_dim; d += 2)
                        {
                            __m256i _pA0 = _mm256_set1_epi32(((const int*)pA0)[0]);
                            __m256i _pA1 = _mm256_set1_epi32(((const int*)pA1)[0]);
                            __m256i _pB = _mm256_loadu_si256((const __m256i*)pK);
#if _MSC_VER
                            // msvc crash here  --- nihui
                            __m256 _pA00 = _mm256_castsi256_ps(_mm256_slli_epi32(_pA0, 16));
                            __m256 _pA10 = _mm256_castsi256_ps(_mm256_slli_epi32(_pA1, 16));
                            __m256 _pB0 = _mm256_castsi256_ps(_mm256_slli_epi32(_pB, 16));
                            __m256 _pA01 = _mm256_castsi256_ps(_mm256_and_si256(_pA0, _mask));
                            __m256 _pA11 = _mm256_castsi256_ps(_mm256_and_si256(_pA1, _mask));
                            __m256 _pB1 = _mm256_castsi256_ps(_mm256_and_si256(_pB, _mask));
                            _sum0 = _mm256_fmadd_ps(_pA00, _pB0, _sum0);
                            _sum1 = _mm256_fmadd_ps(_pA10, _pB0, _sum1);
                            _sum2 = _mm256_fmadd_ps(_pA01, _pB1, _sum2);
                            _sum3 = _mm256_fmadd_ps(_pA11, _pB1, _sum3);
#else
                            _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA0, (__m256bh)_pB);
                            _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA1, (__m256bh)_pB);
#endif
                            pA0 += 2;
                            pA1 += 2;
                            pK += NR * 2;
                        }
#if _MSC_VER
                        _sum0 = _mm256_add_ps(_sum0, _sum2);
                        _sum1 = _mm256_add_ps(_sum1, _sum3);
#endif
                        pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            __m256 _k = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pK));
                            _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(bfloat16_to_float32(*pA0++)), _k, _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(bfloat16_to_float32(*pA1++)), _k, _sum1);
                            pK += NR;
                        }
                        __m256 _scale = _mm256_set1_ps(scale);
                        _sum0 = _mm256_mul_ps(_sum0, _scale);
                        _sum1 = _mm256_mul_ps(_sum1, _scale);
                        if (packed_maskptr0)
                        {
                            _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)packed_maskptr0)));
                            _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)packed_maskptr1)));
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
                        const unsigned short* pA0 = pQ0;
                        const unsigned short* pA1 = pQ1;
                        const unsigned short* pK = key_panel + j;
                        int d = 0;
#if __AVX512BF16__
#if _MSC_VER
                        __m128 _sum2 = _mm_setzero_ps();
                        __m128 _sum3 = _mm_setzero_ps();
                        __m128i _mask = _mm_set1_epi32(0xffff0000);
#endif
                        pK = key_panel + j * 2;
                        for (; d + 1 < head_dim; d += 2)
                        {
                            __m128i _pA0 = _mm_set1_epi32(((const int*)pA0)[0]);
                            __m128i _pA1 = _mm_set1_epi32(((const int*)pA1)[0]);
                            __m128i _pB = _mm_loadu_si128((const __m128i*)pK);
#if _MSC_VER
                            // msvc crash here  --- nihui
                            __m128 _pA00 = _mm_castsi128_ps(_mm_slli_epi32(_pA0, 16));
                            __m128 _pA10 = _mm_castsi128_ps(_mm_slli_epi32(_pA1, 16));
                            __m128 _pB0 = _mm_castsi128_ps(_mm_slli_epi32(_pB, 16));
                            __m128 _pA01 = _mm_castsi128_ps(_mm_and_si128(_pA0, _mask));
                            __m128 _pA11 = _mm_castsi128_ps(_mm_and_si128(_pA1, _mask));
                            __m128 _pB1 = _mm_castsi128_ps(_mm_and_si128(_pB, _mask));
                            _sum0 = _mm_fmadd_ps(_pA00, _pB0, _sum0);
                            _sum1 = _mm_fmadd_ps(_pA10, _pB0, _sum1);
                            _sum2 = _mm_fmadd_ps(_pA01, _pB1, _sum2);
                            _sum3 = _mm_fmadd_ps(_pA11, _pB1, _sum3);
#else
                            _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA0, (__m128bh)_pB);
                            _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA1, (__m128bh)_pB);
#endif
                            pA0 += 2;
                            pA1 += 2;
                            pK += NR * 2;
                        }
#if _MSC_VER
                        _sum0 = _mm_add_ps(_sum0, _sum2);
                        _sum1 = _mm_add_ps(_sum1, _sum3);
#endif
                        pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            __m128 _k = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK));
                            _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(bfloat16_to_float32(*pA0++)), _k, _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(bfloat16_to_float32(*pA1++)), _k, _sum1);
                            pK += NR;
                        }
                        __m128 _scale = _mm_set1_ps(scale);
                        _sum0 = _mm_mul_ps(_sum0, _scale);
                        _sum1 = _mm_mul_ps(_sum1, _scale);
                        if (packed_maskptr0)
                        {
                            _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)packed_maskptr0)));
                            _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)packed_maskptr1)));
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
                        float sum00 = 0.f;
                        float sum01 = 0.f;
                        float sum10 = 0.f;
                        float sum11 = 0.f;
                        const unsigned short* pK = key_panel + j;
                        const unsigned short* pA0 = pQ0;
                        const unsigned short* pA1 = pQ1;
                        int d = 0;
#if __AVX512BF16__
                        pK = key_panel + j * 2;
                        __m128 _sum = _mm_setzero_ps();
                        for (; d + 1 < head_dim; d += 2)
                        {
                            __m128i _q0 = _mm_set1_epi32(((const int*)pA0)[0]);
                            __m128i _q1 = _mm_set1_epi32(((const int*)pA1)[0]);
                            __m128i _k0 = _mm_set1_epi32(((const int*)pK)[0]);
                            __m128i _k1 = _mm_set1_epi32(((const int*)pK)[1]);
                            __m128i _q = _mm_unpacklo_epi64(_q0, _q1);
                            __m128i _k = _mm_unpacklo_epi32(_k0, _k1);
                            _sum = _mm_dpbf16_ps(_sum, (__m128bh)_q, (__m128bh)_k);
                            pA0 += 2;
                            pA1 += 2;
                            pK += NR * 2;
                        }
                        sum00 = _mm_cvtss_f32(_sum);
                        sum01 = _mm_cvtss_f32(_mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(1, 1, 1, 1)));
                        sum10 = _mm_cvtss_f32(_mm_movehl_ps(_sum, _sum));
                        sum11 = _mm_cvtss_f32(_mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(3, 3, 3, 3)));
                        pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            const float k0 = bfloat16_to_float32(pK[0]);
                            const float k1 = bfloat16_to_float32(pK[1]);
                            const float qv0 = bfloat16_to_float32(*pA0++);
                            const float qv1 = bfloat16_to_float32(*pA1++);
                            sum00 += qv0 * k0;
                            sum01 += qv0 * k1;
                            sum10 += qv1 * k0;
                            sum11 += qv1 * k1;
                            pK += NR;
                        }
                        sum00 = sum00 * scale + (packed_maskptr0 ? bfloat16_to_float32(packed_maskptr0[0]) : 0.f);
                        sum01 = sum01 * scale + (packed_maskptr0 ? bfloat16_to_float32(packed_maskptr0[1]) : 0.f);
                        sum10 = sum10 * scale + (packed_maskptr1 ? bfloat16_to_float32(packed_maskptr1[0]) : 0.f);
                        sum11 = sum11 * scale + (packed_maskptr1 ? bfloat16_to_float32(packed_maskptr1[1]) : 0.f);
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
                        float sum0 = 0.f;
                        float sum1 = 0.f;
                        const unsigned short* pK = key_panel + j;
                        const unsigned short* pA0 = pQ0;
                        const unsigned short* pA1 = pQ1;
                        int d = 0;
#if __AVX512BF16__
                        pK = key_panel + j * 2;
                        __m128 _sum = _mm_setzero_ps();
                        for (; d + 1 < head_dim; d += 2)
                        {
                            __m128i _q0 = _mm_set1_epi32(((const int*)pA0)[0]);
                            __m128i _q1 = _mm_set1_epi32(((const int*)pA1)[0]);
                            __m128i _q = _mm_unpacklo_epi64(_q0, _q1);
                            _sum = _mm_dpbf16_ps(_sum, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK)[0]));
                            pA0 += 2;
                            pA1 += 2;
                            pK += NR * 2;
                        }
                        sum0 = _mm_cvtss_f32(_sum);
                        sum1 = _mm_cvtss_f32(_mm_movehl_ps(_sum, _sum));
                        pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            const float v = bfloat16_to_float32(*pK);
                            sum0 += bfloat16_to_float32(*pA0++) * v;
                            sum1 += bfloat16_to_float32(*pA1++) * v;
                            pK += NR;
                        }
                        sum0 = sum0 * scale + (packed_maskptr0 ? bfloat16_to_float32(*packed_maskptr0) : 0.f);
                        sum1 = sum1 * scale + (packed_maskptr1 ? bfloat16_to_float32(*packed_maskptr1) : 0.f);
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
                for (int j = 0; j < max_jj; j++)
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
                if (!computation_value_head.empty())
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
                else
                {
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR;
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
                            const unsigned short* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV0 = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m512 _v = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV0));
                                    _out0 = _mm512_fmadd_ps(_v, _mm512_set1_ps(*scoreptr0++), _out0);
                                    _out1 = _mm512_fmadd_ps(_v, _mm512_set1_ps(*scoreptr1++), _out1);
                                    pV0 += value_panel_width;
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
                            const unsigned short* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV0 = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m256 _v = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV0));
                                    _out0 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(*scoreptr0++), _out0);
                                    _out1 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(*scoreptr1++), _out1);
                                    pV0 += value_panel_width;
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
                            const unsigned short* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV0 = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m128 _v = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV0));
                                    _out0 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(*scoreptr0++), _out0);
                                    _out1 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(*scoreptr1++), _out1);
                                    pV0 += value_panel_width;
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
                            const unsigned short* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV0 = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    const float v0 = bfloat16_to_float32(pV0[0]);
                                    const float v1 = bfloat16_to_float32(pV0[1]);
                                    out00 += *scoreptr0 * v0;
                                    out01 += *scoreptr0++ * v1;
                                    out10 += *scoreptr1 * v0;
                                    out11 += *scoreptr1++ * v1;
                                    pV0 += value_panel_width;
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
                            const unsigned short* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV0 = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    const float v = bfloat16_to_float32(*pV0);
                                    out00 += *scoreptr0++ * v;
                                    out10 += *scoreptr1++ * v;
                                    pV0 += value_panel_width;
                                }
                                pV_panel += (size_t)NR * value_dim;
                            }
                            *outptr0++ = out00;
                            *outptr1++ = out10;
                        }
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
            const unsigned short* qptr = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const unsigned short* packed_mask0 = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + n : 0;
            float block_max = -FLT_MAX;

            // qk
            {
                const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)n * head_dim;
                float* score_panel = scoreptr;
                const unsigned short* packed_maskptr = packed_mask0;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; k + 15 < max_nn; k += 16)
                    {
                        const unsigned short* pA = qptr;
                        const unsigned short* pK = key_panel + k;
                        __m512 _sum = _mm512_setzero_ps();
                        int d = 0;
#if __AVX512BF16__
                        pK = key_panel + k * 2;
                        for (; d + 1 < head_dim; d += 2)
                        {
                            __m512i _key = _mm512_loadu_si512((const __m512i*)pK);
                            __m512i _query = _mm512_set1_epi32(((const int*)pA)[0]);
                            _sum = _mm512_dpbf16_ps(_sum, (__m512bh)_key, (__m512bh)_query);
                            pA += 2;
                            pK += NR * 2;
                        }
                        pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            __m512 _key = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pK));
                            _sum = _mm512_fmadd_ps(_key, _mm512_set1_ps(bfloat16_to_float32(*pA++)), _sum);
                            pK += NR;
                        }
                        _sum = _mm512_mul_ps(_sum, _mm512_set1_ps(scale));
                        if (packed_maskptr)
                        {
                            _sum = _mm512_add_ps(_sum, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)packed_maskptr)));
                            packed_maskptr += 16;
                        }
                        _mm512_storeu_ps(score_panel, _sum);
                        score_panel += 16;
                        block_max = std::max(block_max, _mm512_reduce_max_ps(_sum));
                    }
#endif // __AVX512F__
                    for (; k + 7 < max_nn; k += 8)
                    {
                        const unsigned short* pA = qptr;
                        const unsigned short* pK = key_panel + k;
                        __m256 _sum = _mm256_setzero_ps();
                        int d = 0;
#if __AVX512BF16__
                        pK = key_panel + k * 2;
                        for (; d + 1 < head_dim; d += 2)
                        {
                            __m256i _key = _mm256_loadu_si256((const __m256i*)pK);
                            __m256i _query = _mm256_set1_epi32(((const int*)pA)[0]);
                            _sum = _mm256_dpbf16_ps(_sum, (__m256bh)_key, (__m256bh)_query);
                            pA += 2;
                            pK += NR * 2;
                        }
                        pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            __m256 _key = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pK));
                            _sum = _mm256_comp_fmadd_ps(_key, _mm256_set1_ps(bfloat16_to_float32(*pA++)), _sum);
                            pK += NR;
                        }
                        _sum = _mm256_mul_ps(_sum, _mm256_set1_ps(scale));
                        if (packed_maskptr)
                        {
                            _sum = _mm256_add_ps(_sum, bfloat2float_avx(_mm_loadu_si128((const __m128i*)packed_maskptr)));
                            packed_maskptr += 8;
                        }
                        _mm256_storeu_ps(score_panel, _sum);
                        score_panel += 8;
                        block_max = std::max(block_max, _mm256_reduce_max_ps(_sum));
                    }
#endif // __AVX__
                    for (; k + 3 < max_nn; k += 4)
                    {
                        const unsigned short* pA = qptr;
                        const unsigned short* pK = key_panel + k;
                        __m128 _sum = _mm_setzero_ps();
                        int d = 0;
#if __AVX512BF16__
                        pK = key_panel + k * 2;
                        for (; d + 1 < head_dim; d += 2)
                        {
                            __m128i _key = _mm_loadu_si128((const __m128i*)pK);
                            __m128i _query = _mm_set1_epi32(((const int*)pA)[0]);
                            _sum = _mm_dpbf16_ps(_sum, (__m128bh)_key, (__m128bh)_query);
                            pA += 2;
                            pK += NR * 2;
                        }
                        pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            __m128 _key = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK));
                            _sum = _mm_comp_fmadd_ps(_key, _mm_set1_ps(bfloat16_to_float32(*pA++)), _sum);
                            pK += NR;
                        }
                        _sum = _mm_mul_ps(_sum, _mm_set1_ps(scale));
                        if (packed_maskptr)
                        {
                            _sum = _mm_add_ps(_sum, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)packed_maskptr)));
                            packed_maskptr += 4;
                        }
                        _mm_storeu_ps(score_panel, _sum);
                        score_panel += 4;
                        block_max = std::max(block_max, _mm_reduce_max_ps(_sum));
                    }
#endif // __SSE2__
                    for (; k < max_nn; k++)
                    {
                        const unsigned short* pK = key_panel + k;
                        const unsigned short* pA = qptr;
                        float sum = 0.f;
                        int d = 0;
#if __AVX512BF16__
                        pK = key_panel + k * 2;
                        for (; d + 1 < head_dim; d += 2)
                        {
                            sum += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pK[0]);
                            sum += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pK[1]);
                            pA += 2;
                            pK += NR * 2;
                        }
                        pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
                            sum += bfloat16_to_float32(*pA++) * bfloat16_to_float32(*pK);
                            pK += NR;
                        }
                        sum *= scale;
                        if (packed_maskptr)
                            sum += bfloat16_to_float32(*packed_maskptr++);
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
                for (int j = 0; j < max_jj; j++)
                {
                    *pS = expf(*pS - m_new);
                    sum += *pS++;
                }
                m = m_new;
                l = l * alpha + sum;
            }

            // pv
            {
                if (!computation_value_head.empty())
                {
                    const float* value_panel = computation_value_head.row(n / TILE_N);
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        float* outptr = outT_ptr + (size_t)ii * value_dim + d;
                        for (int lane = 0; lane < value_panel_width; lane++)
                        {
                            float sum = *outptr * alpha;
                            const float* pS = scoreptr;
                            const float* pV = value_panel + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                sum += *pS++ * *pV;
                                pV += value_panel_width;
                            }
                            *outptr++ = sum;
                        }
                        value_panel += (size_t)max_jj * value_panel_width;
                        d += value_panel_width;
                    }
                }
                else
                {
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        float* outptr = outT_ptr + (size_t)ii * value_dim + d;
                        int lane = 0;
#if __SSE2__
                        for (; lane + 3 < value_panel_width; lane += 4)
                        {
                            __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _mm_set1_ps(alpha));
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV = value_panel;
                                const float* pS = scoreptr + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    _out = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV)), _mm_set1_ps(*pS++), _out);
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            _mm_storeu_ps(outptr, _out);
                            outptr += 4;
                        }
#endif // __SSE2__
                        for (; lane < value_panel_width; lane++)
                        {
                            float sum = *outptr * alpha;
                            const unsigned short* value_panel = (const unsigned short*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const unsigned short* pV = value_panel;
                                const float* pS = scoreptr + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    sum += *pS++ * bfloat16_to_float32(*pV);
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            *outptr++ = sum;
                        }
                        d += value_panel_width;
                    }
                }
            }
        }

        lptr[ii] = l;
    }
}

static int sdpa_prefill_packed_bf16s(const Mat& query, const Mat& packed_key, const Mat& packed_value, const Mat& value, const Mat& attn_mask, Mat& top_blob, float scale, const Option& opt)
{
    const int head_dim = query.w;
    const int value_dim = packed_value.empty() ? value.w : packed_value.w;
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
    int value_pack_reuse = 4;
#if __SSE2__
#if __AVX__
    value_pack_reuse = 3;
#endif // __AVX__
#endif // __SSE2__
    if (value_dim < 32)
        value_pack_reuse += 2;
    const int value_reuse = num_mblocks * num_query_heads_per_kv_head;
    const bool use_fp32_value = value_reuse >= value_pack_reuse;
    int TILE_N = sdpa_prefill_get_optimal_tile_n(head_dim, value_dim, key_seqlen, 2, 2, use_fp32_value ? 4 : 2, attn_mask.empty() ? 0 : 2, TILE_M);
    TILE_N = std::max(NR, (TILE_N + NR - 1) / NR * NR);
    const int num_key_blocks = (key_seqlen + TILE_N - 1) / TILE_N;

    Mat packed_value_bf16s = packed_value;
    Mat packed_value_fp32;
    if (use_fp32_value)
    {
        packed_value_fp32.create(value_dim * TILE_N, num_key_blocks, num_kv_heads, 4u, opt.workspace_allocator);
        if (packed_value_fp32.empty())
            return -100;

        const int num_pack_chunks = std::min(num_key_blocks, std::max(1, (nT + num_kv_heads - 1) / num_kv_heads));

        #pragma omp parallel for num_threads(nT)
        for (int task_id = 0; task_id < num_kv_heads * num_pack_chunks; task_id++)
        {
            const int g = task_id / num_pack_chunks;
            const int chunk_id = task_id % num_pack_chunks;
            const int block_begin_id = chunk_id * num_key_blocks / num_pack_chunks;
            const int block_end_id = (chunk_id + 1) * num_key_blocks / num_pack_chunks;
            const Mat packed_value_head = packed_value.empty() ? Mat() : packed_value.channel(g);
            const Mat value_head = value.empty() ? Mat() : value.channel(g);
            Mat packed_value_fp32_head = packed_value_fp32.channel(g);

            for (int block_id = block_begin_id; block_id < block_end_id; block_id++)
            {
                const int block_begin = block_id * TILE_N;
                const int block_seqlen = std::min(TILE_N, key_seqlen - block_begin);
                Mat packed_value_fp32_tile(value_dim * TILE_N, packed_value_fp32_head.row(block_id), 4u);

                if (!packed_value.empty())
                    sdpa_pack_value_tile_bf16s_fp32(packed_value_head, packed_value_fp32_tile, block_begin, 0, block_seqlen, block_seqlen);
                else
                    sdpa_pack_value_tile_bf16s_to_fp32(value_head, packed_value_fp32_tile, block_begin, 0, block_seqlen, block_seqlen);
            }
        }
    }
    else if (packed_value.empty())
    {
        const int capacity = (key_seqlen + NR - 1) / NR * NR;
        packed_value_bf16s.create(value_dim, capacity, num_kv_heads, 2u, 1, opt.workspace_allocator);
        if (packed_value_bf16s.empty())
            return -100;

        packed_value_bf16s.h = key_seqlen;

        const int num_panels = (key_seqlen + NR - 1) / NR;
        const int pack_nT = value.h >= NR ? nT : 1;
        const int num_pack_chunks = std::min(num_panels, std::max(1, (pack_nT + num_kv_heads - 1) / num_kv_heads));

        #pragma omp parallel for num_threads(pack_nT)
        for (int task_id = 0; task_id < num_kv_heads * num_pack_chunks; task_id++)
        {
            const int g = task_id / num_pack_chunks;
            const int chunk_id = task_id % num_pack_chunks;
            const int panel_begin_id = chunk_id * num_panels / num_pack_chunks;
            const int panel_end_id = (chunk_id + 1) * num_panels / num_pack_chunks;
            const Mat value_head = value.channel(g);
            Mat packed_value_head = packed_value_bf16s.channel(g);

            for (int panel_id = panel_begin_id; panel_id < panel_end_id; panel_id++)
            {
                const int panel_begin = panel_id * NR;
                const int n_end = std::min(key_seqlen, panel_begin + NR);
                Mat packed_value_tile(value.w * NR, (unsigned short*)packed_value_head + (size_t)panel_id * value.w * NR, 2u);

                sdpa_pack_value_tile_bf16s(value_head, packed_value_tile, panel_begin, 0, n_end - panel_begin);
            }
        }
    }

    const int query_workspace_size = TILE_M * head_dim;
    const int score_workspace_size = TILE_M * TILE_N;
    const int out_workspace_size = TILE_M * value_dim;
    const int l_workspace_size = TILE_M;
    const int workspace_data_size = score_workspace_size + out_workspace_size + l_workspace_size;
    const int workspace_size = workspace_data_size + (query_workspace_size + 1) / 2;
    Mat workspace(workspace_size, 1, nT, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    Mat packed_mask;
    if (!attn_mask.empty())
    {
        const int num_mask_heads = attn_mask.dims == 3 ? attn_mask.c : 1;
        packed_mask.create(key_seqlen * TILE_M, num_mblocks, num_mask_heads, 2u, opt.workspace_allocator);
        if (packed_mask.empty())
            return -100;

        sdpa_pack_mask_bf16s(attn_mask, packed_mask, TILE_M, opt);
    }

#pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int q = task_id / num_mblocks;
        const int i0 = task_id % num_mblocks * TILE_M;
        const int max_ii = std::min(query_seqlen - i0, TILE_M);
        const int g = q / num_query_heads_per_kv_head;

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        float* workspace_ptr = workspace_tile;
        Mat scoreT(score_workspace_size, workspace_ptr, 4u);
        Mat outT(out_workspace_size, workspace_ptr + score_workspace_size, 4u);
        Mat lT(l_workspace_size, workspace_ptr + score_workspace_size + out_workspace_size, 4u);
        Mat queryT(query_workspace_size, (unsigned short*)(workspace_ptr + workspace_data_size), 2u);

        const Mat query_head = query.channel(q);
        const Mat packed_key_head = packed_key.channel(g);
        const Mat packed_value_head = packed_value_bf16s.empty() ? Mat() : packed_value_bf16s.channel(g);
        const Mat computation_value_head = packed_value_fp32.empty() ? Mat() : packed_value_fp32.channel(g);
        Mat packed_mask_tile;
        if (!packed_mask.empty())
        {
            Mat packed_mask_head = packed_mask.channel(packed_mask.c > 1 ? q : 0);
            packed_mask_tile = Mat(key_seqlen * TILE_M, packed_mask_head.row<unsigned short>(task_id % num_mblocks), 2u);
        }

        sdpa_pack_query_bf16s(query_head, queryT, i0, max_ii, query_head.w * query_head.elempack);

        sdpa_attention_tile_bf16s(queryT, Mat(), packed_key_head, Mat(), packed_value_head, computation_value_head, Mat(), 0, packed_mask_tile, scoreT, outT, lT, max_ii, scale);

        Mat top_blob_head = top_blob.channel(q);
        sdpa_store_output_tile(outT, lT, top_blob_head, i0, max_ii, top_blob_head.w * top_blob_head.elempack);
    }

    return 0;
}

static int sdpa_prefill_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
        return sdpa_prefill_bf16s_avx512bf16(query, key, value, attn_mask_blob, top_blob, scale, opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx2())
        return sdpa_prefill_bf16s_avx2(query, key, value, attn_mask_blob, top_blob, scale, opt);
#endif

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

    Mat packed_key(key.w, capacity, key.c, 2u, 1, opt.workspace_allocator);
    if (packed_key.empty())
        return -100;

    packed_key.h = key.h;

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
        Mat packed_key_head = packed_key.channel(g);

        for (int panel_id = panel_begin_id; panel_id < panel_end_id; panel_id++)
        {
            const int n_begin = panel_id * panel_width;
            const int n_end = std::min(key.h, n_begin + panel_width);
            Mat packed_key_tile(key.w * panel_width, (unsigned short*)packed_key_head + (size_t)panel_id * key.w * panel_width, 2u);
            sdpa_pack_key_tile_bf16s(key_head, packed_key_tile, n_begin, 0, n_end - n_begin);
        }
    }

    return sdpa_prefill_packed_bf16s(query, packed_key, Mat(), value, attn_mask_blob, top_blob, scale, opt);
}

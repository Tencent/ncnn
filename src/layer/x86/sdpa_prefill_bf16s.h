// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void sdpa_pack_query_bf16s_avx512bf16(const Mat& query_head, Mat& queryT, int i, int max_ii);
void sdpa_pack_key_tile_bf16s_avx512bf16(const Mat& key, Mat& packed_key, int src_begin, int dst_begin, int max_seqlen);
void sdpa_prefill_packed_tile_bf16s_avx512bf16(const Mat& queryT, const Mat& packed_key_head, const Mat& packed_value_head, const Mat& packed_value_fp32_head, const Mat& maskT, Mat& scoreT, Mat& outT, Mat& stateT, int max_ii, int n_begin, int n_end, float scale);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
void sdpa_pack_value_tile_bf16s_fp32_avx2(const Mat& packed_value, Mat& packed_value_fp32, int src_begin, int dst_begin, int max_seqlen, int dst_seqlen);
void sdpa_pack_value_tile_bf16s_to_fp32_avx2(const Mat& value, Mat& packed_value_fp32, int src_begin, int dst_begin, int max_seqlen, int dst_seqlen);
void sdpa_prefill_packed_tile_bf16s_avx2(const Mat& queryT, const Mat& packed_key_head, const Mat& packed_value_head, const Mat& packed_value_fp32_head, const Mat& maskT, Mat& scoreT, Mat& outT, Mat& stateT, int max_ii, int n_begin, int n_end, float scale);
#endif

static void sdpa_pack_query_bf16s(const Mat& query_head, Mat& queryT, int i, int max_ii)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        sdpa_pack_query_bf16s_avx512bf16(query_head, queryT, i, max_ii);
        return;
    }
#endif

    const int head_dim = query_head.w;
    unsigned short* pp = queryT;
    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const int i0 = i + ii;
        const unsigned short* qptr0 = query_head.row<const unsigned short>(i0);
        const unsigned short* qptr1 = query_head.row<const unsigned short>(i0 + 1);
        const unsigned short* qptr2 = query_head.row<const unsigned short>(i0 + 2);
        const unsigned short* qptr3 = query_head.row<const unsigned short>(i0 + 3);
        const unsigned short* qptr4 = query_head.row<const unsigned short>(i0 + 4);
        const unsigned short* qptr5 = query_head.row<const unsigned short>(i0 + 5);
        const unsigned short* qptr6 = query_head.row<const unsigned short>(i0 + 6);
        const unsigned short* qptr7 = query_head.row<const unsigned short>(i0 + 7);
        const unsigned short* qptr8 = query_head.row<const unsigned short>(i0 + 8);
        const unsigned short* qptr9 = query_head.row<const unsigned short>(i0 + 9);
        const unsigned short* qptra = query_head.row<const unsigned short>(i0 + 10);
        const unsigned short* qptrb = query_head.row<const unsigned short>(i0 + 11);
        const unsigned short* qptrc = query_head.row<const unsigned short>(i0 + 12);
        const unsigned short* qptrd = query_head.row<const unsigned short>(i0 + 13);
        const unsigned short* qptre = query_head.row<const unsigned short>(i0 + 14);
        const unsigned short* qptrf = query_head.row<const unsigned short>(i0 + 15);

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
            pp += 256;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = qptr0[d];
            pp[1] = qptr0[d + 1];
            pp[2] = qptr1[d];
            pp[3] = qptr1[d + 1];
            pp[4] = qptr2[d];
            pp[5] = qptr2[d + 1];
            pp[6] = qptr3[d];
            pp[7] = qptr3[d + 1];
            pp[8] = qptr4[d];
            pp[9] = qptr4[d + 1];
            pp[10] = qptr5[d];
            pp[11] = qptr5[d + 1];
            pp[12] = qptr6[d];
            pp[13] = qptr6[d + 1];
            pp[14] = qptr7[d];
            pp[15] = qptr7[d + 1];
            pp[16] = qptr8[d];
            pp[17] = qptr8[d + 1];
            pp[18] = qptr9[d];
            pp[19] = qptr9[d + 1];
            pp[20] = qptra[d];
            pp[21] = qptra[d + 1];
            pp[22] = qptrb[d];
            pp[23] = qptrb[d + 1];
            pp[24] = qptrc[d];
            pp[25] = qptrc[d + 1];
            pp[26] = qptrd[d];
            pp[27] = qptrd[d + 1];
            pp[28] = qptre[d];
            pp[29] = qptre[d + 1];
            pp[30] = qptrf[d];
            pp[31] = qptrf[d + 1];
            pp += 32;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pp[0] = qptr0[d];
            pp[1] = qptr1[d];
            pp[2] = qptr2[d];
            pp[3] = qptr3[d];
            pp[4] = qptr4[d];
            pp[5] = qptr5[d];
            pp[6] = qptr6[d];
            pp[7] = qptr7[d];
            pp[8] = qptr8[d];
            pp[9] = qptr9[d];
            pp[10] = qptra[d];
            pp[11] = qptrb[d];
            pp[12] = qptrc[d];
            pp[13] = qptrd[d];
            pp[14] = qptre[d];
            pp[15] = qptrf[d];
            pp += 16;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const int i0 = i + ii;
        const unsigned short* qptr0 = query_head.row<const unsigned short>(i0);
        const unsigned short* qptr1 = query_head.row<const unsigned short>(i0 + 1);
        const unsigned short* qptr2 = query_head.row<const unsigned short>(i0 + 2);
        const unsigned short* qptr3 = query_head.row<const unsigned short>(i0 + 3);
        const unsigned short* qptr4 = query_head.row<const unsigned short>(i0 + 4);
        const unsigned short* qptr5 = query_head.row<const unsigned short>(i0 + 5);
        const unsigned short* qptr6 = query_head.row<const unsigned short>(i0 + 6);
        const unsigned short* qptr7 = query_head.row<const unsigned short>(i0 + 7);

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
            pp += 64;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = qptr0[d];
            pp[1] = qptr0[d + 1];
            pp[2] = qptr1[d];
            pp[3] = qptr1[d + 1];
            pp[4] = qptr2[d];
            pp[5] = qptr2[d + 1];
            pp[6] = qptr3[d];
            pp[7] = qptr3[d + 1];
            pp[8] = qptr4[d];
            pp[9] = qptr4[d + 1];
            pp[10] = qptr5[d];
            pp[11] = qptr5[d + 1];
            pp[12] = qptr6[d];
            pp[13] = qptr6[d + 1];
            pp[14] = qptr7[d];
            pp[15] = qptr7[d + 1];
            pp += 16;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pp[0] = qptr0[d];
            pp[1] = qptr1[d];
            pp[2] = qptr2[d];
            pp[3] = qptr3[d];
            pp[4] = qptr4[d];
            pp[5] = qptr5[d];
            pp[6] = qptr6[d];
            pp[7] = qptr7[d];
            pp += 8;
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int i0 = i + ii;
        const unsigned short* qptr0 = query_head.row<const unsigned short>(i0);
        const unsigned short* qptr1 = query_head.row<const unsigned short>(i0 + 1);
        const unsigned short* qptr2 = query_head.row<const unsigned short>(i0 + 2);
        const unsigned short* qptr3 = query_head.row<const unsigned short>(i0 + 3);

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
            _mm_storeu_si128((__m128i*)pp, _r0);
            _mm_storeu_si128((__m128i*)(pp + 8), _r1);
            pp += 16;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = qptr0[d];
            pp[1] = qptr0[d + 1];
            pp[2] = qptr1[d];
            pp[3] = qptr1[d + 1];
            pp[4] = qptr2[d];
            pp[5] = qptr2[d + 1];
            pp[6] = qptr3[d];
            pp[7] = qptr3[d + 1];
            pp += 8;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pp[0] = qptr0[d];
            pp[1] = qptr1[d];
            pp[2] = qptr2[d];
            pp[3] = qptr3[d];
            pp += 4;
        }
    }
#endif // __SSE2__
    for (; ii < max_ii; ii++)
    {
        const unsigned short* qptr = query_head.row<const unsigned short>(i + ii);
        memcpy(pp, qptr, (size_t)head_dim * sizeof(unsigned short));
        pp += head_dim;
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
        Mat maskT = packed_mask_head.row_range(mblock_id, 1);
        unsigned short* pp = maskT;

        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            const unsigned short* p0 = mask_head.row<const unsigned short>(i0 + ii);
            const unsigned short* p1 = mask_head.row<const unsigned short>(i0 + ii + 1);
            const unsigned short* p2 = mask_head.row<const unsigned short>(i0 + ii + 2);
            const unsigned short* p3 = mask_head.row<const unsigned short>(i0 + ii + 3);
            const unsigned short* p4 = mask_head.row<const unsigned short>(i0 + ii + 4);
            const unsigned short* p5 = mask_head.row<const unsigned short>(i0 + ii + 5);
            const unsigned short* p6 = mask_head.row<const unsigned short>(i0 + ii + 6);
            const unsigned short* p7 = mask_head.row<const unsigned short>(i0 + ii + 7);
            const unsigned short* p8 = mask_head.row<const unsigned short>(i0 + ii + 8);
            const unsigned short* p9 = mask_head.row<const unsigned short>(i0 + ii + 9);
            const unsigned short* pa = mask_head.row<const unsigned short>(i0 + ii + 10);
            const unsigned short* pb = mask_head.row<const unsigned short>(i0 + ii + 11);
            const unsigned short* pc = mask_head.row<const unsigned short>(i0 + ii + 12);
            const unsigned short* pd = mask_head.row<const unsigned short>(i0 + ii + 13);
            const unsigned short* pe = mask_head.row<const unsigned short>(i0 + ii + 14);
            const unsigned short* pf = mask_head.row<const unsigned short>(i0 + ii + 15);

            int j = 0;
            for (; j + 15 < mask_head.w; j += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)p1);
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)p2);
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)p3);
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)p4);
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)p5);
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)p6);
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)p7);
                __m256i _r8 = _mm256_loadu_si256((const __m256i*)p8);
                __m256i _r9 = _mm256_loadu_si256((const __m256i*)p9);
                __m256i _ra = _mm256_loadu_si256((const __m256i*)pa);
                __m256i _rb = _mm256_loadu_si256((const __m256i*)pb);
                __m256i _rc = _mm256_loadu_si256((const __m256i*)pc);
                __m256i _rd = _mm256_loadu_si256((const __m256i*)pd);
                __m256i _re = _mm256_loadu_si256((const __m256i*)pe);
                __m256i _rf = _mm256_loadu_si256((const __m256i*)pf);

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
                pp += 256;
            }
            for (; j < mask_head.w; j++)
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
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = mask_head.row<const unsigned short>(i0 + ii);
            const unsigned short* p1 = mask_head.row<const unsigned short>(i0 + ii + 1);
            const unsigned short* p2 = mask_head.row<const unsigned short>(i0 + ii + 2);
            const unsigned short* p3 = mask_head.row<const unsigned short>(i0 + ii + 3);
            const unsigned short* p4 = mask_head.row<const unsigned short>(i0 + ii + 4);
            const unsigned short* p5 = mask_head.row<const unsigned short>(i0 + ii + 5);
            const unsigned short* p6 = mask_head.row<const unsigned short>(i0 + ii + 6);
            const unsigned short* p7 = mask_head.row<const unsigned short>(i0 + ii + 7);

            int j = 0;
            for (; j + 7 < mask_head.w; j += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _r2 = _mm_loadu_si128((const __m128i*)p2);
                __m128i _r3 = _mm_loadu_si128((const __m128i*)p3);
                __m128i _r4 = _mm_loadu_si128((const __m128i*)p4);
                __m128i _r5 = _mm_loadu_si128((const __m128i*)p5);
                __m128i _r6 = _mm_loadu_si128((const __m128i*)p6);
                __m128i _r7 = _mm_loadu_si128((const __m128i*)p7);

                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                _mm_storeu_si128((__m128i*)(pp + 16), _r2);
                _mm_storeu_si128((__m128i*)(pp + 24), _r3);
                _mm_storeu_si128((__m128i*)(pp + 32), _r4);
                _mm_storeu_si128((__m128i*)(pp + 40), _r5);
                _mm_storeu_si128((__m128i*)(pp + 48), _r6);
                _mm_storeu_si128((__m128i*)(pp + 56), _r7);

                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
                p4 += 8;
                p5 += 8;
                p6 += 8;
                p7 += 8;
                pp += 64;
            }
            for (; j < mask_head.w; j++)
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
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = mask_head.row<const unsigned short>(i0 + ii);
            const unsigned short* p1 = mask_head.row<const unsigned short>(i0 + ii + 1);
            const unsigned short* p2 = mask_head.row<const unsigned short>(i0 + ii + 2);
            const unsigned short* p3 = mask_head.row<const unsigned short>(i0 + ii + 3);

            int j = 0;
            for (; j + 3 < mask_head.w; j += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)p1);
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)p2);
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)p3);

                __m128i _tmp0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _tmp1 = _mm_unpacklo_epi16(_r2, _r3);
                _r0 = _mm_unpacklo_epi32(_tmp0, _tmp1);
                _r1 = _mm_unpackhi_epi32(_tmp0, _tmp1);

                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);

                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                pp += 16;
            }
            for (; j < mask_head.w; j++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp += 4;
            }
        }
#endif // __SSE2__
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = mask_head.row<const unsigned short>(i0 + ii);
            memcpy(pp, p0, (size_t)mask_head.w * sizeof(unsigned short));
            pp += mask_head.w;
        }
    }
}

// packed_key[token_panel][head_dim][token_lane] in bf16
static void sdpa_pack_key_tile_bf16s(const Mat& key, Mat& packed_key, int src_begin, int dst_begin, int max_seqlen)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        sdpa_pack_key_tile_bf16s_avx512bf16(key, packed_key, src_begin, dst_begin, max_seqlen);
        return;
    }
#endif

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
    unsigned short* panel = packed_key;
    int j = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
    for (; j + 15 < max_seqlen; j += 16)
    {
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);
        const unsigned short* p1 = p0 + head_dim;
        const unsigned short* p2 = p1 + head_dim;
        const unsigned short* p3 = p2 + head_dim;
        const unsigned short* p4 = p3 + head_dim;
        const unsigned short* p5 = p4 + head_dim;
        const unsigned short* p6 = p5 + head_dim;
        const unsigned short* p7 = p6 + head_dim;
        const unsigned short* p8 = p7 + head_dim;
        const unsigned short* p9 = p8 + head_dim;
        const unsigned short* pa = p9 + head_dim;
        const unsigned short* pb = pa + head_dim;
        const unsigned short* pc = pb + head_dim;
        const unsigned short* pd = pc + head_dim;
        const unsigned short* pe = pd + head_dim;
        const unsigned short* pf = pe + head_dim;

#if __AVX512BF16__
        unsigned short* pp = panel + (token_lane + j) * 2;
#else
        unsigned short* pp = panel + token_lane + j;
#endif // __AVX512BF16__
        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
            __m256i _r1 = _mm256_loadu_si256((const __m256i*)p1);
            __m256i _r2 = _mm256_loadu_si256((const __m256i*)p2);
            __m256i _r3 = _mm256_loadu_si256((const __m256i*)p3);
            __m256i _r4 = _mm256_loadu_si256((const __m256i*)p4);
            __m256i _r5 = _mm256_loadu_si256((const __m256i*)p5);
            __m256i _r6 = _mm256_loadu_si256((const __m256i*)p6);
            __m256i _r7 = _mm256_loadu_si256((const __m256i*)p7);
            __m256i _r8 = _mm256_loadu_si256((const __m256i*)p8);
            __m256i _r9 = _mm256_loadu_si256((const __m256i*)p9);
            __m256i _ra = _mm256_loadu_si256((const __m256i*)pa);
            __m256i _rb = _mm256_loadu_si256((const __m256i*)pb);
            __m256i _rc = _mm256_loadu_si256((const __m256i*)pc);
            __m256i _rd = _mm256_loadu_si256((const __m256i*)pd);
            __m256i _re = _mm256_loadu_si256((const __m256i*)pe);
            __m256i _rf = _mm256_loadu_si256((const __m256i*)pf);

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
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];
            pp[16] = p8[0];
            pp[17] = p8[1];
            pp[18] = p9[0];
            pp[19] = p9[1];
            pp[20] = pa[0];
            pp[21] = pa[1];
            pp[22] = pb[0];
            pp[23] = pb[1];
            pp[24] = pc[0];
            pp[25] = pc[1];
            pp[26] = pd[0];
            pp[27] = pd[1];
            pp[28] = pe[0];
            pp[29] = pe[1];
            pp[30] = pf[0];
            pp[31] = pf[1];

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
            pp += panel_width * 2;
        }
        if (d < head_dim)
        {
            unsigned short* pptr = panel + (size_t)d * panel_width + token_lane + j;
            pptr[0] = *p0;
            pptr[1] = *p1;
            pptr[2] = *p2;
            pptr[3] = *p3;
            pptr[4] = *p4;
            pptr[5] = *p5;
            pptr[6] = *p6;
            pptr[7] = *p7;
            pptr[8] = *p8;
            pptr[9] = *p9;
            pptr[10] = *pa;
            pptr[11] = *pb;
            pptr[12] = *pc;
            pptr[13] = *pd;
            pptr[14] = *pe;
            pptr[15] = *pf;
        }
#else
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
#endif // __AVX512BF16__
    }
#endif // __AVX512F__
    for (; j + 7 < max_seqlen; j += 8)
    {
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);
        const unsigned short* p1 = p0 + head_dim;
        const unsigned short* p2 = p1 + head_dim;
        const unsigned short* p3 = p2 + head_dim;
        const unsigned short* p4 = p3 + head_dim;
        const unsigned short* p5 = p4 + head_dim;
        const unsigned short* p6 = p5 + head_dim;
        const unsigned short* p7 = p6 + head_dim;

#if __AVX512BF16__
        unsigned short* pp = panel + (token_lane + j) * 2;
#else
        unsigned short* pp = panel + token_lane + j;
#endif // __AVX512BF16__
        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _r1 = _mm_loadu_si128((const __m128i*)p1);
            __m128i _r2 = _mm_loadu_si128((const __m128i*)p2);
            __m128i _r3 = _mm_loadu_si128((const __m128i*)p3);
            __m128i _r4 = _mm_loadu_si128((const __m128i*)p4);
            __m128i _r5 = _mm_loadu_si128((const __m128i*)p5);
            __m128i _r6 = _mm_loadu_si128((const __m128i*)p6);
            __m128i _r7 = _mm_loadu_si128((const __m128i*)p7);
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
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
            pp += panel_width * 8;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];

            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
            pp += panel_width * 2;
        }
        if (d < head_dim)
        {
            unsigned short* pptr = panel + (size_t)d * panel_width + token_lane + j;
            pptr[0] = *p0;
            pptr[1] = *p1;
            pptr[2] = *p2;
            pptr[3] = *p3;
            pptr[4] = *p4;
            pptr[5] = *p5;
            pptr[6] = *p6;
            pptr[7] = *p7;
        }
#else
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
#endif // __AVX512BF16__
    }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; j + 3 < max_seqlen; j += 4)
    {
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);
        const unsigned short* p1 = p0 + head_dim;
        const unsigned short* p2 = p1 + head_dim;
        const unsigned short* p3 = p2 + head_dim;

#if __AVX512BF16__
        unsigned short* pp = panel + (token_lane + j) * 2;
#else
        unsigned short* pp = panel + token_lane + j;
#endif // __AVX512BF16__
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)p1);
            __m128i _r2 = _mm_loadl_epi64((const __m128i*)p2);
            __m128i _r3 = _mm_loadl_epi64((const __m128i*)p3);
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
            p1 += 4;
            p2 += 4;
            p3 += 4;
            pp += panel_width * 4;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];

            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            pp += panel_width * 2;
        }
        if (d < head_dim)
        {
            unsigned short* pptr = panel + (size_t)d * panel_width + token_lane + j;
            pptr[0] = *p0;
            pptr[1] = *p1;
            pptr[2] = *p2;
            pptr[3] = *p3;
        }
#else
        for (; d < head_dim; d++)
        {
            pp[0] = *p0++;
            pp[1] = *p1++;
            pp[2] = *p2++;
            pp[3] = *p3++;
            pp += panel_width;
        }
#endif // __AVX512BF16__
    }
#endif // __SSE2__
    for (; j < max_seqlen; j++)
    {
        const unsigned short* p0 = key.row<const unsigned short>(src_begin + j);
#if __AVX512BF16__
        unsigned short* pp = panel + (token_lane + j) * 2;
        int d = 0;
        for (; d + 1 < head_dim; d += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            p0 += 2;
            pp += panel_width * 2;
        }
        if (d < head_dim)
            panel[(size_t)d * panel_width + token_lane + j] = *p0;
#else
        unsigned short* pp = panel + token_lane + j;
        for (int d = 0; d < head_dim; d++)
        {
            *pp = *p0++;
            pp += panel_width;
        }
#endif // __AVX512BF16__
    }
}

// packed_value[token_panel][value_panel][token_lane][value_lane] in bf16
static void sdpa_pack_value_tile_bf16s(const Mat& value, Mat& packed_value, int src_begin, int dst_begin, int max_seqlen)
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

// packed_value_fp32[key_block][value_panel][token][value_lane]
static void sdpa_pack_value_tile_bf16s_fp32(const Mat& packed_value, Mat& packed_value_fp32, int src_begin, int dst_begin, int max_seqlen, int dst_seqlen)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx2())
    {
        sdpa_pack_value_tile_bf16s_fp32_avx2(packed_value, packed_value_fp32, src_begin, dst_begin, max_seqlen, dst_seqlen);
        return;
    }
#endif

    const int value_dim = packed_value.w;
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
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx2())
    {
        sdpa_pack_value_tile_bf16s_to_fp32_avx2(value, packed_value_fp32, src_begin, dst_begin, max_seqlen, dst_seqlen);
        return;
    }
#endif

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

static void sdpa_prefill_packed_tile_bf16s(const Mat& queryT, const Mat& packed_key_head, const Mat& packed_value_head, const Mat& packed_value_fp32_head, const Mat& maskT, Mat& scoreT, Mat& outT, Mat& stateT, int max_ii, int n_begin, int n_end, float scale)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        sdpa_prefill_packed_tile_bf16s_avx512bf16(queryT, packed_key_head, packed_value_head, packed_value_fp32_head, maskT, scoreT, outT, stateT, max_ii, n_begin, n_end, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx2())
    {
        sdpa_prefill_packed_tile_bf16s_avx2(queryT, packed_key_head, packed_value_head, packed_value_fp32_head, maskT, scoreT, outT, stateT, max_ii, n_begin, n_end, scale);
        return;
    }
#endif

    const int head_dim = packed_key_head.w;
    const int key_seqlen = packed_key_head.h;
    const int TILE_M = stateT.w / 2;
    const int TILE_N = scoreT.w / TILE_M;
    const int value_dim = outT.w / TILE_M;
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

    const unsigned short* queryT_ptr = queryT;
    float* scoreT_ptr = scoreT;
    float* outT_ptr = outT;
    float* mT = stateT;
    float* lT = mT + TILE_M;
    const unsigned short* maskT_ptr = maskT;

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
            const unsigned short* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const unsigned short* pM = maskT_ptr ? maskT_ptr + (size_t)ii * key_seqlen + (size_t)n * 16 : 0;
            __m512 _block_max = _mm512_set1_ps(-FLT_MAX);

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)(n + jj) * head_dim;

                if (max_nn == 16)
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
                    int d = 0;
#if __AVX512BF16__
                    const int* pK = (const int*)key_panel;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        const __m512i _q = _mm512_loadu_si512((const __m512i*)pA);
                        _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[0]));
                        _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[1]));
                        _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[2]));
                        _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[3]));
                        _sum4 = _mm512_dpbf16_ps(_sum4, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[4]));
                        _sum5 = _mm512_dpbf16_ps(_sum5, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[5]));
                        _sum6 = _mm512_dpbf16_ps(_sum6, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[6]));
                        _sum7 = _mm512_dpbf16_ps(_sum7, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[7]));
                        _sum8 = _mm512_dpbf16_ps(_sum8, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[8]));
                        _sum9 = _mm512_dpbf16_ps(_sum9, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[9]));
                        _suma = _mm512_dpbf16_ps(_suma, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[10]));
                        _sumb = _mm512_dpbf16_ps(_sumb, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[11]));
                        _sumc = _mm512_dpbf16_ps(_sumc, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[12]));
                        _sumd = _mm512_dpbf16_ps(_sumd, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[13]));
                        _sume = _mm512_dpbf16_ps(_sume, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[14]));
                        _sumf = _mm512_dpbf16_ps(_sumf, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK[15]));
                        pA += 32;
                        pK += 16;
                    }
                    const unsigned short* pK_tail = (const unsigned short*)pK;
#else
                    const unsigned short* pK = key_panel;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        const __m512 _q = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
#if __AVX512BF16__
                        const unsigned short* pB = pK_tail;
#else
                        const unsigned short* pB = pK;
#endif // __AVX512BF16__
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[3])), _sum3);
                        _sum4 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[4])), _sum4);
                        _sum5 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[5])), _sum5);
                        _sum6 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[6])), _sum6);
                        _sum7 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[7])), _sum7);
                        _sum8 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[8])), _sum8);
                        _sum9 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[9])), _sum9);
                        _suma = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[10])), _suma);
                        _sumb = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[11])), _sumb);
                        _sumc = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[12])), _sumc);
                        _sumd = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[13])), _sumd);
                        _sume = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[14])), _sume);
                        _sumf = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pB[15])), _sumf);
                        pA += 16;
#if __AVX512BF16__
                        pK_tail += 16;
#else
                        pK += 16;
#endif // __AVX512BF16__
                    }

                    const __m512 _scale = _mm512_set1_ps(scale);
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
                        _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pM)));
                        _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 16))));
                        _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 32))));
                        _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 48))));
                        _sum4 = _mm512_add_ps(_sum4, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 64))));
                        _sum5 = _mm512_add_ps(_sum5, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 80))));
                        _sum6 = _mm512_add_ps(_sum6, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 96))));
                        _sum7 = _mm512_add_ps(_sum7, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 112))));
                        _sum8 = _mm512_add_ps(_sum8, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 128))));
                        _sum9 = _mm512_add_ps(_sum9, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 144))));
                        _suma = _mm512_add_ps(_suma, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 160))));
                        _sumb = _mm512_add_ps(_sumb, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 176))));
                        _sumc = _mm512_add_ps(_sumc, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 192))));
                        _sumd = _mm512_add_ps(_sumd, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 208))));
                        _sume = _mm512_add_ps(_sume, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 224))));
                        _sumf = _mm512_add_ps(_sumf, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 240))));
                    }
                    _mm512_storeu_ps(scoreptr + (size_t)jj * 16, _sum0);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 1) * 16, _sum1);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 2) * 16, _sum2);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 3) * 16, _sum3);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 4) * 16, _sum4);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 5) * 16, _sum5);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 6) * 16, _sum6);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 7) * 16, _sum7);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 8) * 16, _sum8);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 9) * 16, _sum9);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 10) * 16, _suma);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 11) * 16, _sumb);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 12) * 16, _sumc);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 13) * 16, _sumd);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 14) * 16, _sume);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + 15) * 16, _sumf);
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(_sum0, _sum1), _mm512_max_ps(_sum2, _sum3)), _mm512_max_ps(_mm512_max_ps(_sum4, _sum5), _mm512_max_ps(_sum6, _sum7))));
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(_sum8, _sum9), _mm512_max_ps(_suma, _sumb)), _mm512_max_ps(_mm512_max_ps(_sumc, _sumd), _mm512_max_ps(_sume, _sumf))));
                }
                else
                {
                    for (int j = 0; j < max_nn; j++)
                    {
                        __m512 _sum = _mm512_setzero_ps();
                        const unsigned short* pA = pQ;
                        int d = 0;
#if __AVX512BF16__
                        const int* pK = (const int*)key_panel + j;
                        for (; d + 1 < head_dim; d += 2)
                        {
                            _sum = _mm512_dpbf16_ps(_sum, (__m512bh)_mm512_loadu_si512((const __m512i*)pA), (__m512bh)_mm512_set1_epi32(*pK));
                            pA += 32;
                            pK += 16;
                        }
                        const unsigned short* pK_tail = key_panel + (size_t)d * NR + j;
#else
                        const unsigned short* pK = key_panel + j;
#endif // __AVX512BF16__
                        for (; d < head_dim; d++)
                        {
#if __AVX512BF16__
                            const unsigned short k = *pK_tail;
                            pK_tail += 16;
#else
                            const unsigned short k = *pK;
                            pK += 16;
#endif // __AVX512BF16__
                            _sum = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA)), _mm512_set1_ps(bfloat16_to_float32(k)), _sum);
                            pA += 16;
                        }
                        _sum = _mm512_mul_ps(_sum, _mm512_set1_ps(scale));
                        if (pM)
                            _sum = _mm512_add_ps(_sum, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + (size_t)j * 16))));
                        _mm512_storeu_ps(scoreptr + (size_t)(jj + j) * 16, _sum);
                        _block_max = _mm512_max_ps(_block_max, _sum);
                    }
                }

                if (pM)
                    pM += (size_t)max_nn * 16;
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
                const __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + (size_t)j * 16), _m_new));
                _mm512_storeu_ps(scoreptr + (size_t)j * 16, _p);
                _sum0 = _mm512_add_ps(_sum0, _p);
            }
            _mm512_storeu_ps(mT + ii, _m_new);
            _mm512_storeu_ps(lT + ii, _mm512_add_ps(_mm512_mul_ps(_l, _alpha), _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3))));
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const unsigned short* pM = maskT_ptr ? maskT_ptr + (size_t)ii * key_seqlen + (size_t)n * 8 : 0;
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)(n + jj) * head_dim;
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
                    const unsigned short* pA = pQ;
                    int d = 0;
#if __AVX512BF16__
                    const int* pK = (const int*)key_panel + j;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        const __m256i _q = _mm256_loadu_si256((const __m256i*)pA);
                        _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK[0]));
                        _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK[1]));
                        _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK[2]));
                        _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK[3]));
                        _sum4 = _mm256_dpbf16_ps(_sum4, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK[4]));
                        _sum5 = _mm256_dpbf16_ps(_sum5, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK[5]));
                        _sum6 = _mm256_dpbf16_ps(_sum6, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK[6]));
                        _sum7 = _mm256_dpbf16_ps(_sum7, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK[7]));
                        pA += 16;
                        pK += NR;
                    }
                    const unsigned short* pK_tail = key_panel + (size_t)d * NR + j;
#else
                    const unsigned short* pK = key_panel + j;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        const __m256 _q = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
#if __AVX512BF16__
                        const unsigned short* pB = pK_tail;
#else
                        const unsigned short* pB = pK;
#endif // __AVX512BF16__
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pB[3])), _sum3);
                        _sum4 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pB[4])), _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pB[5])), _sum5);
                        _sum6 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pB[6])), _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pB[7])), _sum7);
                        pA += 8;
#if __AVX512BF16__
                        pK_tail += NR;
#else
                        pK += NR;
#endif // __AVX512BF16__
                    }
                    const __m256 _scale = _mm256_set1_ps(scale);
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
                        _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)j * 8))));
                        _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)(j + 1) * 8))));
                        _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)(j + 2) * 8))));
                        _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)(j + 3) * 8))));
                        _sum4 = _mm256_add_ps(_sum4, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)(j + 4) * 8))));
                        _sum5 = _mm256_add_ps(_sum5, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)(j + 5) * 8))));
                        _sum6 = _mm256_add_ps(_sum6, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)(j + 6) * 8))));
                        _sum7 = _mm256_add_ps(_sum7, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)(j + 7) * 8))));
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
                    if (pM)
                        _sum = _mm256_add_ps(_sum, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + (size_t)j * 8))));
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j) * 8, _sum);
                    _block_max = _mm256_max_ps(_block_max, _sum);
                }

                if (pM)
                    pM += (size_t)max_nn * 8;
            }

            const __m256 _m = _mm256_loadu_ps(mT + ii);
            const __m256 _l = _mm256_loadu_ps(lT + ii);
            const __m256 _m_new = _mm256_max_ps(_m, _block_max);
            const __m256 _alpha_active = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            const __m256 _alpha = _mm256_and_ps(exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new))), _alpha_active);

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
                const __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + (size_t)j * 8), _m_new));
                _mm256_storeu_ps(scoreptr + (size_t)j * 8, _p);
                _sum0 = _mm256_add_ps(_sum0, _p);
            }
            _mm256_storeu_ps(mT + ii, _m_new);
            _mm256_storeu_ps(lT + ii, _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3))));
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const unsigned short* pM = maskT_ptr ? maskT_ptr + (size_t)ii * key_seqlen + (size_t)n * 4 : 0;
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)(n + jj) * head_dim;
                for (int j = 0; j < max_nn; j++)
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
                    if (pM)
                        _sum = _mm_add_ps(_sum, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + (size_t)j * 4))));
                    _mm_storeu_ps(scoreptr + (size_t)(jj + j) * 4, _sum);
                    _block_max = _mm_max_ps(_block_max, _sum);
                }

                if (pM)
                    pM += (size_t)max_nn * 4;
            }

            const __m128 _m = _mm_loadu_ps(mT + ii);
            const __m128 _l = _mm_loadu_ps(lT + ii);
            const __m128 _m_new = _mm_max_ps(_m, _block_max);
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
            _mm_storeu_ps(mT + ii, _m_new);
            _mm_storeu_ps(lT + ii, _mm_add_ps(_mm_mul_ps(_l, _alpha), _mm_add_ps(_mm_add_ps(_sum0, _sum1), _mm_add_ps(_sum2, _sum3))));
        }
#endif // __SSE2__
        for (; ii < max_ii; ii++)
        {
            const unsigned short* qptr = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const unsigned short* mask0 = maskT_ptr ? maskT_ptr + (size_t)ii * key_seqlen + n : 0;
            float block_max = -FLT_MAX;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)packed_key_head + (size_t)(n + jj) * head_dim;
                int k = 0;
#if __SSE2__
                for (; k < max_nn; k += 4)
                {
                    const int max_kk = std::min(4, max_nn - k);
                    const unsigned short* pK = key_panel + k;
                    __m128 _sum = _mm_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        const unsigned short* pK_pair = key_panel + (size_t)d * NR + k * 2;
                        const __m128 _k0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK_pair));
                        const __m128 _k1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pK_pair + 4)));
                        const __m128 _key0 = _mm_shuffle_ps(_k0, _k1, _MM_SHUFFLE(2, 0, 2, 0));
                        const __m128 _key1 = _mm_shuffle_ps(_k0, _k1, _MM_SHUFFLE(3, 1, 3, 1));
                        _sum = _mm_comp_fmadd_ps(_key0, _mm_set1_ps(bfloat16_to_float32(qptr[d])), _sum);
                        _sum = _mm_comp_fmadd_ps(_key1, _mm_set1_ps(bfloat16_to_float32(qptr[d + 1])), _sum);
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK)), _mm_set1_ps(bfloat16_to_float32(qptr[d])), _sum);
                        pK += NR;
                    }
                    _sum = _mm_mul_ps(_sum, _mm_set1_ps(scale));
                    if (mask0)
                    {
                        const unsigned short* pM = mask0 + jj + k;
                        _sum = _mm_add_ps(_sum, _mm_set_ps(max_kk > 3 ? bfloat16_to_float32(pM[3]) : 0.f, max_kk > 2 ? bfloat16_to_float32(pM[2]) : 0.f, max_kk > 1 ? bfloat16_to_float32(pM[1]) : 0.f, bfloat16_to_float32(pM[0])));
                    }
                    _mm_storeu_ps(scoreptr + jj + k, _sum);
                    for (int kk = 0; kk < max_kk; kk++)
                        block_max = std::max(block_max, scoreptr[jj + k + kk]);
                }
#endif // __SSE2__
                for (; k < max_nn; k++)
                {
                    const unsigned short* pK = key_panel + k;
                    float sum = 0.f;
                    for (int d = 0; d < head_dim; d++)
                    {
                        sum += bfloat16_to_float32(qptr[d]) * bfloat16_to_float32(*pK);
                        pK += NR;
                    }
                    sum *= scale;
                    if (mask0)
                        sum += bfloat16_to_float32(mask0[jj + k]);
                    scoreptr[jj + k] = sum;
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

        const int value_block = packed_value_fp32_head.empty() ? NR : max_jj;
        for (int jj = 0; jj < max_jj; jj += value_block)
        {
            const int max_nn = std::min(value_block, max_jj - jj);
            const unsigned short* value_panel = packed_value_head.empty() ? 0 : (const unsigned short*)packed_value_head + (size_t)(n + jj) * value_dim;
            const float* value_panel_fp32 = packed_value_fp32_head.empty() ? 0 : packed_value_fp32_head.row(n / TILE_N);

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
                    if (value_panel_width == 16)
                    {
                        __m512 _out0 = _mm512_loadu_ps(outptr);
                        __m512 _out1 = _mm512_loadu_ps(outptr + 16);
                        __m512 _out2 = _mm512_loadu_ps(outptr + 32);
                        __m512 _out3 = _mm512_loadu_ps(outptr + 48);
                        __m512 _out4 = _mm512_loadu_ps(outptr + 64);
                        __m512 _out5 = _mm512_loadu_ps(outptr + 80);
                        __m512 _out6 = _mm512_loadu_ps(outptr + 96);
                        __m512 _out7 = _mm512_loadu_ps(outptr + 112);
                        __m512 _out8 = _mm512_loadu_ps(outptr + 128);
                        __m512 _out9 = _mm512_loadu_ps(outptr + 144);
                        __m512 _outa = _mm512_loadu_ps(outptr + 160);
                        __m512 _outb = _mm512_loadu_ps(outptr + 176);
                        __m512 _outc = _mm512_loadu_ps(outptr + 192);
                        __m512 _outd = _mm512_loadu_ps(outptr + 208);
                        __m512 _oute = _mm512_loadu_ps(outptr + 224);
                        __m512 _outf = _mm512_loadu_ps(outptr + 240);
                        const unsigned short* pV = value_panel ? value_panel + (size_t)d * NR : 0;
                        const float* pV_fp32 = value_panel_fp32;
                        const float* pS = scoreptr;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m512 _p = _mm512_loadu_ps(pS);
                            const __m512 _v = pV_fp32 ? _mm512_loadu_ps(pV_fp32) : bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV));
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
                            if (pV_fp32)
                                pV_fp32 += 16;
                            else
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
                    {
                        for (int lane = 0; lane < value_panel_width; lane++)
                        {
                            __m512 _out = _mm512_loadu_ps(outptr + (size_t)lane * 16);
                            const float* pS = scoreptr;
                            const unsigned short* pV = value_panel ? value_panel + (size_t)d * NR + lane : 0;
                            const float* pV_fp32 = value_panel_fp32 ? value_panel_fp32 + lane : 0;
                            for (int j = 0; j < max_nn; j++)
                            {
                                const float v = pV_fp32 ? *pV_fp32 : bfloat16_to_float32(*pV);
                                _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(v), _out);
                                pS += 16;
                                if (pV_fp32)
                                    pV_fp32 += value_panel_width;
                                else
                                    pV += value_panel_width;
                            }
                            _mm512_storeu_ps(outptr + (size_t)lane * 16, _out);
                        }
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
                        const unsigned short* pV = value_panel ? value_panel + (size_t)d * NR + lane : 0;
                        const float* pV_fp32 = value_panel_fp32 ? value_panel_fp32 + lane : 0;
                        const float* pS = scoreptr;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m256 _p = _mm256_loadu_ps(pS);
                            const __m256 _v = pV_fp32 ? _mm256_loadu_ps(pV_fp32) : bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV));
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
                            if (pV_fp32)
                                pV_fp32 += value_panel_width;
                            else
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
                        const float* pS = scoreptr;
                        const unsigned short* pV = value_panel ? value_panel + (size_t)d * NR + lane : 0;
                        const float* pV_fp32 = value_panel_fp32 ? value_panel_fp32 + lane : 0;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const float v = pV_fp32 ? *pV_fp32 : bfloat16_to_float32(*pV);
                            _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(v), _out);
                            pS += 8;
                            if (pV_fp32)
                                pV_fp32 += value_panel_width;
                            else
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
                        float* outptr0 = outptr + (size_t)lane * 4;
                        __m128 _out0 = _mm_loadu_ps(outptr0);
                        __m128 _out1 = _mm_loadu_ps(outptr0 + 4);
                        __m128 _out2 = _mm_loadu_ps(outptr0 + 8);
                        __m128 _out3 = _mm_loadu_ps(outptr0 + 12);
                        const float* pS = scoreptr;
                        const unsigned short* pV = value_panel ? value_panel + (size_t)d * NR + lane : 0;
                        const float* pV_fp32 = value_panel_fp32 ? value_panel_fp32 + lane : 0;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m128 _p = _mm_loadu_ps(pS);
                            const __m128 _v = pV_fp32 ? _mm_loadu_ps(pV_fp32) : bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV));
                            _out0 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                            _out1 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                            _out2 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                            _out3 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                            pS += 4;
                            if (pV_fp32)
                                pV_fp32 += value_panel_width;
                            else
                                pV += value_panel_width;
                        }
                        _mm_storeu_ps(outptr0, _out0);
                        _mm_storeu_ps(outptr0 + 4, _out1);
                        _mm_storeu_ps(outptr0 + 8, _out2);
                        _mm_storeu_ps(outptr0 + 12, _out3);
                    }
                    for (; lane < value_panel_width; lane++)
                    {
                        __m128 _out = _mm_loadu_ps(outptr + (size_t)lane * 4);
                        const float* pS = scoreptr;
                        const unsigned short* pV = value_panel ? value_panel + (size_t)d * NR + lane : 0;
                        const float* pV_fp32 = value_panel_fp32 ? value_panel_fp32 + lane : 0;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const float v = pV_fp32 ? *pV_fp32 : bfloat16_to_float32(*pV);
                            _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(v), _out);
                            pS += 4;
                            if (pV_fp32)
                                pV_fp32 += value_panel_width;
                            else
                                pV += value_panel_width;
                        }
                        _mm_storeu_ps(outptr + (size_t)lane * 4, _out);
                    }
                }
#endif // __SSE2__
                for (; ii < max_ii; ii++)
                {
                    const float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N + jj;
                    float* outptr = outT_ptr + (size_t)ii * value_dim + d;
                    int lane = 0;
#if __SSE2__
                    for (; lane + 3 < value_panel_width; lane += 4)
                    {
                        __m128 _out = _mm_loadu_ps(outptr + lane);
                        const unsigned short* pV0 = value_panel ? value_panel + (size_t)d * NR + lane : 0;
                        const float* pV0_fp32 = value_panel_fp32 ? value_panel_fp32 + lane : 0;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m128 _v = pV0_fp32 ? _mm_loadu_ps(pV0_fp32) : bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV0));
                            _out = _mm_comp_fmadd_ps(_v, _mm_set1_ps(scoreptr[j]), _out);
                            if (pV0_fp32)
                                pV0_fp32 += value_panel_width;
                            else
                                pV0 += value_panel_width;
                        }
                        _mm_storeu_ps(outptr + lane, _out);
                    }
#endif // __SSE2__
                    for (; lane < value_panel_width; lane++)
                    {
                        float sum = outptr[lane];
                        const unsigned short* pV0 = value_panel ? value_panel + (size_t)d * NR + lane : 0;
                        const float* pV0_fp32 = value_panel_fp32 ? value_panel_fp32 + lane : 0;
                        for (int j = 0; j < max_nn; j++)
                        {
                            sum += scoreptr[j] * (pV0_fp32 ? *pV0_fp32 : bfloat16_to_float32(*pV0));
                            if (pV0_fp32)
                                pV0_fp32 += value_panel_width;
                            else
                                pV0 += value_panel_width;
                        }
                        outptr[lane] = sum;
                    }
                }

                if (value_panel_fp32)
                    value_panel_fp32 += (size_t)max_nn * value_panel_width;

                d += value_panel_width;
            }
        }
    }

}

static int sdpa_prefill_packed_bf16s(const Mat& query, const Mat& packed_key, const Mat& packed_value, const Mat& value, int value_begin, const Mat& attn_mask, Mat& top_blob, float scale, const Option& opt)
{
    const int head_dim = query.w;
    const int value_dim = value.w;
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
    int TILE_N = sdpa_prefill_get_optimal_tile_n(head_dim, value_dim, key_seqlen, 2, 2, use_fp32_value ? 4 : 2, attn_mask.empty() ? 0 : 2, TILE_M, num_tasks, nT);
    TILE_N = std::max(NR, (TILE_N + NR - 1) / NR * NR);
    const int num_key_blocks = (key_seqlen + TILE_N - 1) / TILE_N;

    Mat packed_value_bf16s = packed_value;
    Mat packed_value_fp32;
    if (use_fp32_value)
    {
        packed_value_fp32.create(value_dim * TILE_N, num_key_blocks, num_kv_heads, 4u, opt.workspace_allocator);
        if (packed_value_fp32.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_kv_heads * num_key_blocks; task_id++)
        {
            const int g = task_id / num_key_blocks;
            const int block_id = task_id % num_key_blocks;
            const int block_begin = block_id * TILE_N;
            const int block_seqlen = std::min(TILE_N, key_seqlen - block_begin);
            Mat packed_value_fp32_tile = packed_value_fp32.channel(g).row_range(block_id, 1);

            if (block_begin < value_begin)
            {
                const int n_begin = block_begin;
                const int n_end = std::min(value_begin, block_begin + block_seqlen);
                const Mat packed_value_head = packed_value_bf16s.channel(g);
                sdpa_pack_value_tile_bf16s_fp32(packed_value_head, packed_value_fp32_tile, n_begin, 0, n_end - n_begin, block_seqlen);
            }

            if (block_begin + block_seqlen > value_begin)
            {
                const int n_begin = std::max(value_begin, block_begin);
                const int n_end = block_begin + block_seqlen;
                const Mat value_head = value.channel(g);
                sdpa_pack_value_tile_bf16s_to_fp32(value_head, packed_value_fp32_tile, n_begin - value_begin, n_begin - block_begin, n_end - n_begin, block_seqlen);
            }
        }
    }
    else if (packed_value_bf16s.empty())
    {
        const int capacity = (key_seqlen + NR - 1) / NR * NR;
        packed_value_bf16s.create(value_dim, capacity, num_kv_heads, 2u, 1, opt.workspace_allocator);
        if (packed_value_bf16s.empty())
            return -100;

        packed_value_bf16s.h = key_seqlen;

        const int num_panels = (key_seqlen + NR - 1) / NR;
        const int pack_nT = value.h >= NR ? opt.num_threads : 1;

        #pragma omp parallel for num_threads(pack_nT)
        for (int task_id = 0; task_id < num_kv_heads * num_panels; task_id++)
        {
            const int g = task_id / num_panels;
            const int panel_id = task_id % num_panels;
            const int panel_begin = panel_id * NR;
            const int n_begin = panel_begin;
            const int n_end = std::min(key_seqlen, panel_begin + NR);
            const Mat value_head = value.channel(g);
            Mat packed_value_head = packed_value_bf16s.channel(g);
            Mat packed_value_tile(value.w * NR, (unsigned short*)packed_value_head + (size_t)panel_id * value.w * NR, 2u);

            sdpa_pack_value_tile_bf16s(value_head, packed_value_tile, n_begin, 0, n_end - n_begin);
        }
    }

    int num_kv_chunks = 1;
    if (num_tasks < nT && num_key_blocks >= 2)
    {
        num_kv_chunks = std::min((nT + num_tasks - 1) / num_tasks, num_key_blocks);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    const int query_workspace_size = TILE_M * head_dim;
    Mat query_workspace(query_workspace_size, 1, nT, 2u, opt.workspace_allocator);
    if (query_workspace.empty())
        return -100;

    const int score_workspace_size = TILE_M * TILE_N;
    const int out_workspace_size = TILE_M * value_dim;
    const int state_workspace_size = TILE_M * 2;
    const int workspace_size = score_workspace_size + out_workspace_size + state_workspace_size;
    Mat workspace(workspace_size, 1, nT, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    Mat packed_query;
    if (num_kv_chunks > 1)
    {
        packed_query.create(query_workspace_size, 1, num_tasks, 2u, opt.workspace_allocator);
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
            sdpa_pack_query_bf16s(query_head, queryT, i0, max_ii);
        }
    }

    Mat packed_mask;
    if (!attn_mask.empty())
    {
        const int num_mask_heads = attn_mask.dims == 3 ? attn_mask.c : 1;
        packed_mask.create(key_seqlen * TILE_M, num_mblocks, num_mask_heads, 2u, opt.workspace_allocator);
        if (packed_mask.empty())
            return -100;

        sdpa_pack_mask_bf16s(attn_mask, packed_mask, TILE_M, opt);
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

        Mat queryT = query_workspace.channel(get_omp_thread_num());
        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        Mat scoreT = workspace_tile.range(0, score_workspace_size);
        Mat outT = workspace_tile.range(score_workspace_size, out_workspace_size);
        Mat stateT = workspace_tile.range(score_workspace_size + out_workspace_size, state_workspace_size);
        float* outT_ptr = outT;
        float* mT = stateT;
        float* lT = mT + TILE_M;

        const Mat query_head = query.channel(q);
        const Mat packed_key_head = packed_key.channel(g);
        const Mat packed_value_head = packed_value_bf16s.empty() ? Mat() : packed_value_bf16s.channel(g);
        const Mat packed_value_fp32_head = packed_value_fp32.empty() ? Mat() : packed_value_fp32.channel(g);
        Mat maskT;
        if (!packed_mask.empty())
        {
            Mat packed_mask_head = packed_mask.channel(packed_mask.c > 1 ? q : 0);
            maskT = packed_mask_head.row_range(task_id % num_mblocks, 1);
        }

        if (!packed_query.empty())
            queryT = packed_query.channel(task_id);
        else
            sdpa_pack_query_bf16s(query_head, queryT, i0, max_ii);

        sdpa_prefill_packed_tile_bf16s(queryT, packed_key_head, packed_value_head, packed_value_fp32_head, maskT, scoreT, outT, stateT, max_ii, n_begin, n_end, scale);

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

static int sdpa_prefill_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
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

    Mat packed_key(key.w, capacity, key.c, 2u, 1, opt.workspace_allocator);
    if (packed_key.empty())
        return -100;

    packed_key.h = key.h;

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
        Mat packed_key_head = packed_key.channel(g);
        Mat packed_key_tile(key.w * panel_width, (unsigned short*)packed_key_head + (size_t)panel_id * key.w * panel_width, 2u);

        sdpa_pack_key_tile_bf16s(key_head, packed_key_tile, n_begin, 0, n_end - n_begin);
    }

    return sdpa_prefill_packed_bf16s(query, packed_key, Mat(), value, 0, attn_mask_blob, top_blob, scale, opt);
}

static int sdpa_kvcache_bf16s(const Mat& query, const Mat& past_key, const Mat& past_value, const Mat& cur_key, const Mat& cur_value, Mat& cached_key, Mat& cached_value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
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
        Mat packed_key_tile(cur_key.w * panel_width, (unsigned short*)packed_key_head + (size_t)panel_id * cur_key.w * panel_width, 2u);
        Mat packed_value_tile(cur_value.w * panel_width, (unsigned short*)packed_value_head + (size_t)panel_id * cur_value.w * panel_width, 2u);

        sdpa_pack_key_tile_bf16s(key_head, packed_key_tile, n_begin - past_seqlen, n_begin - panel_begin, n_end - n_begin);
        sdpa_pack_value_tile_bf16s(value_head, packed_value_tile, n_begin - past_seqlen, n_begin - panel_begin, n_end - n_begin);
    }

    if (query.h == 1)
        return sdpa_decode_kvcache_bf16s(query, cached_key, cached_value, attn_mask_blob, top_blob, scale, opt);

    return sdpa_prefill_packed_bf16s(query, cached_key, cached_value, cur_value, past_seqlen, attn_mask_blob, top_blob, scale, opt);
}

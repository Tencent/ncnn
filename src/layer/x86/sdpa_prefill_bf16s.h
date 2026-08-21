// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
int sdpa_prefill_bf16s_avx512bf16(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
int sdpa_prefill_bf16s_avx2(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt);
#endif

// packed_key[block][key_panel][head_dim][key_lane] in bf16
// avx512bf16 pairs adjacent head_dim values in each key lane
static void sdpa_pack_key_bf16s(const Mat& key, Mat& packed_key, int block_n, const Option& opt)
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
        const unsigned short* key_base = key_head.row<const unsigned short>(n);
        unsigned short* pp = packed_key.channel(g).row<unsigned short>(block_id);

        const int max_jj = std::min(block_n, key_seqlen - n);
        int j = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
        for (; j + 15 < max_jj; j += 16)
        {
            const unsigned short* p0 = key_base + (size_t)j * head_dim;
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

            int k = 0;
            for (; k + 15 < head_dim; k += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)(p0 + k));
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p1 + k));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p2 + k));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p3 + k));
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)(p4 + k));
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)(p5 + k));
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)(p6 + k));
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)(p7 + k));
                __m256i _r8 = _mm256_loadu_si256((const __m256i*)(p8 + k));
                __m256i _r9 = _mm256_loadu_si256((const __m256i*)(p9 + k));
                __m256i _ra = _mm256_loadu_si256((const __m256i*)(pa + k));
                __m256i _rb = _mm256_loadu_si256((const __m256i*)(pb + k));
                __m256i _rc = _mm256_loadu_si256((const __m256i*)(pc + k));
                __m256i _rd = _mm256_loadu_si256((const __m256i*)(pd + k));
                __m256i _re = _mm256_loadu_si256((const __m256i*)(pe + k));
                __m256i _rf = _mm256_loadu_si256((const __m256i*)(pf + k));

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
            for (; k + 1 < head_dim; k += 2)
            {
                pp[0] = p0[k];
                pp[1] = p0[k + 1];
                pp[2] = p1[k];
                pp[3] = p1[k + 1];
                pp[4] = p2[k];
                pp[5] = p2[k + 1];
                pp[6] = p3[k];
                pp[7] = p3[k + 1];
                pp[8] = p4[k];
                pp[9] = p4[k + 1];
                pp[10] = p5[k];
                pp[11] = p5[k + 1];
                pp[12] = p6[k];
                pp[13] = p6[k + 1];
                pp[14] = p7[k];
                pp[15] = p7[k + 1];
                pp[16] = p8[k];
                pp[17] = p8[k + 1];
                pp[18] = p9[k];
                pp[19] = p9[k + 1];
                pp[20] = pa[k];
                pp[21] = pa[k + 1];
                pp[22] = pb[k];
                pp[23] = pb[k + 1];
                pp[24] = pc[k];
                pp[25] = pc[k + 1];
                pp[26] = pd[k];
                pp[27] = pd[k + 1];
                pp[28] = pe[k];
                pp[29] = pe[k + 1];
                pp[30] = pf[k];
                pp[31] = pf[k + 1];
                pp += 32;
            }
#endif // __AVX512BF16__
            for (; k < head_dim; k++)
            {
                pp[0] = p0[k];
                pp[1] = p1[k];
                pp[2] = p2[k];
                pp[3] = p3[k];
                pp[4] = p4[k];
                pp[5] = p5[k];
                pp[6] = p6[k];
                pp[7] = p7[k];
                pp[8] = p8[k];
                pp[9] = p9[k];
                pp[10] = pa[k];
                pp[11] = pb[k];
                pp[12] = pc[k];
                pp[13] = pd[k];
                pp[14] = pe[k];
                pp[15] = pf[k];
                pp += 16;
            }
        }
#endif // __AVX512F__
        for (; j + 7 < max_jj; j += 8)
        {
            const unsigned short* p0 = key_base + (size_t)j * head_dim;
            const unsigned short* p1 = p0 + head_dim;
            const unsigned short* p2 = p1 + head_dim;
            const unsigned short* p3 = p2 + head_dim;
            const unsigned short* p4 = p3 + head_dim;
            const unsigned short* p5 = p4 + head_dim;
            const unsigned short* p6 = p5 + head_dim;
            const unsigned short* p7 = p6 + head_dim;

            int k = 0;
            for (; k + 7 < head_dim; k += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)(p0 + k));
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p1 + k));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p2 + k));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p3 + k));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p4 + k));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p5 + k));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p6 + k));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p7 + k));

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
            for (; k + 1 < head_dim; k += 2)
            {
                pp[0] = p0[k];
                pp[1] = p0[k + 1];
                pp[2] = p1[k];
                pp[3] = p1[k + 1];
                pp[4] = p2[k];
                pp[5] = p2[k + 1];
                pp[6] = p3[k];
                pp[7] = p3[k + 1];
                pp[8] = p4[k];
                pp[9] = p4[k + 1];
                pp[10] = p5[k];
                pp[11] = p5[k + 1];
                pp[12] = p6[k];
                pp[13] = p6[k + 1];
                pp[14] = p7[k];
                pp[15] = p7[k + 1];
                pp += 16;
            }
#endif // __AVX512BF16__
            for (; k < head_dim; k++)
            {
                pp[0] = p0[k];
                pp[1] = p1[k];
                pp[2] = p2[k];
                pp[3] = p3[k];
                pp[4] = p4[k];
                pp[5] = p5[k];
                pp[6] = p6[k];
                pp[7] = p7[k];
                pp += 8;
            }
        }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; j + 3 < max_jj; j += 4)
        {
            const unsigned short* p0 = key_base + (size_t)j * head_dim;
            const unsigned short* p1 = p0 + head_dim;
            const unsigned short* p2 = p1 + head_dim;
            const unsigned short* p3 = p2 + head_dim;

            int k = 0;
            for (; k + 3 < head_dim; k += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)(p0 + k));
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p1 + k));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p2 + k));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p3 + k));

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
            for (; k + 1 < head_dim; k += 2)
            {
                pp[0] = p0[k];
                pp[1] = p0[k + 1];
                pp[2] = p1[k];
                pp[3] = p1[k + 1];
                pp[4] = p2[k];
                pp[5] = p2[k + 1];
                pp[6] = p3[k];
                pp[7] = p3[k + 1];
                pp += 8;
            }
#endif // __AVX512BF16__
            for (; k < head_dim; k++)
            {
                pp[0] = p0[k];
                pp[1] = p1[k];
                pp[2] = p2[k];
                pp[3] = p3[k];
                pp += 4;
            }
        }
#endif // __SSE2__
        for (; j < max_jj; j++)
        {
            const unsigned short* p0 = key_base + (size_t)j * head_dim;
            memcpy(pp, p0, (size_t)head_dim * sizeof(unsigned short));
            pp += head_dim;
        }
    }
}

// packed_value uses the fp32 layout consumed by the pv kernels
static void sdpa_pack_value_bf16s(const Mat& value, Mat& packed_value, int block_n, const Option& opt)
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
        const unsigned short* value_base = value_head.row<const unsigned short>(n);
        float* pp = packed_value.channel(g).row(block_id);

        int d = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
        for (; d + 15 < value_dim; d += 16)
        {
            const unsigned short* p0 = value_base + d;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm512_storeu_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
                _mm512_storeu_ps(pp + 16, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + value_dim))));
                _mm512_storeu_ps(pp + 32, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + value_dim * 2))));
                _mm512_storeu_ps(pp + 48, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + value_dim * 3))));
                pp += 64;
                p0 += value_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm512_storeu_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
                pp += 16;
                p0 += value_dim;
            }
        }
#endif // __AVX512F__
        for (; d + 7 < value_dim; d += 8)
        {
            const unsigned short* p0 = value_base + d;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                _mm256_storeu_ps(pp + 8, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + value_dim))));
                _mm256_storeu_ps(pp + 16, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + value_dim * 2))));
                _mm256_storeu_ps(pp + 24, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + value_dim * 3))));
                pp += 32;
                p0 += value_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                pp += 8;
                p0 += value_dim;
            }
        }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; d + 3 < value_dim; d += 4)
        {
            const unsigned short* p0 = value_base + d;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm_storeu_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                _mm_storeu_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + value_dim))));
                _mm_storeu_ps(pp + 8, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + value_dim * 2))));
                _mm_storeu_ps(pp + 12, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + value_dim * 3))));
                pp += 16;
                p0 += value_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm_storeu_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                pp += 4;
                p0 += value_dim;
            }
        }
#endif // __SSE2__
        for (; d < value_dim; d++)
        {
            const unsigned short* p0 = value_base + d;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p0[value_dim]);
                pp[2] = bfloat16_to_float32(p0[value_dim * 2]);
                pp[3] = bfloat16_to_float32(p0[value_dim * 3]);
                pp += 4;
                p0 += value_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                *pp++ = bfloat16_to_float32(*p0);
                p0 += value_dim;
            }
        }
    }
}

// queryT[head_dim][query_lane] in bf16 or fp32
// avx512bf16 pairs adjacent head_dim values in each query lane
static void sdpa_pack_query_bf16s(const Mat& query_head, Mat& queryT, int i, int max_ii)
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
    }
}

static void sdpa_flash_attention_tile_bf16s(const Mat& query, const Mat& key, const Mat& packed_key, const Mat& value, const Mat& packed_value, const Mat& attn_mask_blob, const Mat& packed_mask, Mat& top_blob, float scale, int q, int g, int i0, int max_ii, int n_begin, int n_end, int block_n, int state_stride, const Mat& packed_query, Mat& workspace, Mat& state)
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
        queryT = Mat(head_dim * max_ii, workspace_ptr + (block_n + value_dim) * max_ii, 2u);

        sdpa_pack_query_bf16s(query_head, queryT, i0, max_ii);
    }
    const unsigned short* queryT_base = queryT;

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
        const unsigned short* queryT = queryT_base + ii * head_dim;
        const Mat key_head = key.channel(g);
        const Mat packed_key_head = packed_key.empty() ? Mat() : packed_key.channel(g);
        const Mat value_head = value.channel(g);
        const Mat packed_value_head = packed_value.empty() ? Mat() : packed_value.channel(g);
        const unsigned short* maskT = packed_mask.empty() ? 0 : (const unsigned short*)packed_mask + (size_t)ii * key_seqlen;

        memset(outT, 0, (size_t)value_dim * 16 * sizeof(float));
        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();
        const unsigned short* pM = maskT ? maskT + (size_t)n_begin * 16 : 0;
        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m512 _block_max;
            float* scoreptr = scoreT;
            if (packed_key.empty())
            {
                const unsigned short* key = key_head.row<const unsigned short>(n);
                const __m512 _scale = _mm512_set1_ps(scale);
                __m512 _max = _mm512_set1_ps(-FLT_MAX);
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
                for (; j + 7 < max_jj; j += 8)
                {
                    const unsigned short* pQ = queryT;
                    const unsigned short* pK0 = key + (size_t)j * head_dim;
                    const unsigned short* pK1 = pK0 + head_dim;
                    const unsigned short* pK2 = pK1 + head_dim;
                    const unsigned short* pK3 = pK2 + head_dim;
                    const unsigned short* pK4 = pK3 + head_dim;
                    const unsigned short* pK5 = pK4 + head_dim;
                    const unsigned short* pK6 = pK5 + head_dim;
                    const unsigned short* pK7 = pK6 + head_dim;
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    __m512 _sum4 = _mm512_setzero_ps();
                    __m512 _sum5 = _mm512_setzero_ps();
                    __m512 _sum6 = _mm512_setzero_ps();
                    __m512 _sum7 = _mm512_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _q = _mm512_loadu_si512((const __m512i*)pQ);
                        _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK0)[0]));
                        _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK1)[0]));
                        _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK2)[0]));
                        _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK3)[0]));
                        _sum4 = _mm512_dpbf16_ps(_sum4, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK4)[0]));
                        _sum5 = _mm512_dpbf16_ps(_sum5, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK5)[0]));
                        _sum6 = _mm512_dpbf16_ps(_sum6, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK6)[0]));
                        _sum7 = _mm512_dpbf16_ps(_sum7, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK7)[0]));
                        pQ += 32;
                        pK0 += 2;
                        pK1 += 2;
                        pK2 += 2;
                        pK3 += 2;
                        pK4 += 2;
                        pK5 += 2;
                        pK6 += 2;
                        pK7 += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m512 _q = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pQ));
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK0++)), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK1++)), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK2++)), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK3++)), _sum3);
                        _sum4 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK4++)), _sum4);
                        _sum5 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK5++)), _sum5);
                        _sum6 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK6++)), _sum6);
                        _sum7 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK7++)), _sum7);
                        pQ += 16;
                    }
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
                        _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pM)));
                        _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 16))));
                        _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 32))));
                        _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 48))));
                        _sum4 = _mm512_add_ps(_sum4, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 64))));
                        _sum5 = _mm512_add_ps(_sum5, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 80))));
                        _sum6 = _mm512_add_ps(_sum6, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 96))));
                        _sum7 = _mm512_add_ps(_sum7, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 112))));
                        pM += 128;
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
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j + 3 < max_jj; j += 4)
                {
                    const unsigned short* pQ = queryT;
                    const unsigned short* pK0 = key + (size_t)j * head_dim;
                    const unsigned short* pK1 = pK0 + head_dim;
                    const unsigned short* pK2 = pK1 + head_dim;
                    const unsigned short* pK3 = pK2 + head_dim;
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _q = _mm512_loadu_si512((const __m512i*)pQ);
                        _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK0)[0]));
                        _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK1)[0]));
                        _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK2)[0]));
                        _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK3)[0]));
                        pQ += 32;
                        pK0 += 2;
                        pK1 += 2;
                        pK2 += 2;
                        pK3 += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m512 _q = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pQ));
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK0++)), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK1++)), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK2++)), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(*pK3++)), _sum3);
                        pQ += 16;
                    }
                    _sum0 = _mm512_mul_ps(_sum0, _scale);
                    _sum1 = _mm512_mul_ps(_sum1, _scale);
                    _sum2 = _mm512_mul_ps(_sum2, _scale);
                    _sum3 = _mm512_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pM)));
                        _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 16))));
                        _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 32))));
                        _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 48))));
                        pM += 64;
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
                    const unsigned short* pQ = queryT;
                    const unsigned short* pK = key + (size_t)j * head_dim;
                    __m512 _sum = _mm512_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _q = _mm512_loadu_si512((const __m512i*)pQ);
                        _sum = _mm512_dpbf16_ps(_sum, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK)[0]));
                        pQ += 32;
                        pK += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pQ)), _mm512_set1_ps(bfloat16_to_float32(*pK++)), _sum);
                        pQ += 16;
                    }
                    _sum = _mm512_mul_ps(_sum, _scale);
                    if (pM)
                    {
                        _sum = _mm512_add_ps(_sum, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pM)));
                        pM += 16;
                    }
                    _max = _mm512_max_ps(_max, _sum);
                    _mm512_storeu_ps(scoreptr, _sum);
                    scoreptr += 16;
                }

                _block_max = _max;
            }
            else
            {
                const unsigned short* pK = packed_key_head.row<const unsigned short>(n / block_n);
                const unsigned short* pQ = queryT;
                const __m512 _scale = _mm512_set1_ps(scale);
                __m512 _max = _mm512_set1_ps(-FLT_MAX);
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
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

                    const unsigned short* pA = pQ;
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
                        pM += 256;
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

                    const unsigned short* pA = pQ;
                    int d = 0;
#if __AVX512BF16__
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
                        pK += 16;
                    }
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
                        pK += 8;
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
                        _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pM)));
                        _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 16))));
                        _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 32))));
                        _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 48))));
                        _sum4 = _mm512_add_ps(_sum4, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 64))));
                        _sum5 = _mm512_add_ps(_sum5, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 80))));
                        _sum6 = _mm512_add_ps(_sum6, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 96))));
                        _sum7 = _mm512_add_ps(_sum7, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 112))));
                        pM += 128;
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
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j + 3 < max_jj; j += 4)
                {
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();

                    const unsigned short* pA = pQ;
                    int d = 0;
#if __AVX512BF16__
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
                        pK += 8;
                    }
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
                        pK += 4;
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

                    _sum0 = _mm512_mul_ps(_sum0, _scale);
                    _sum1 = _mm512_mul_ps(_sum1, _scale);
                    _sum2 = _mm512_mul_ps(_sum2, _scale);
                    _sum3 = _mm512_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pM)));
                        _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 16))));
                        _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 32))));
                        _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + 48))));
                        pM += 64;
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
                    const unsigned short* pA = pQ;
                    const unsigned short* pK0 = pK;
                    __m512 _sum = _mm512_setzero_ps();

                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                        _sum = _mm512_dpbf16_ps(_sum, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pK0)[0]));
                        pA += 32;
                        pK0 += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m512 _pA = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                        _sum = _mm512_fmadd_ps(_pA, _mm512_set1_ps(bfloat16_to_float32(pK0[0])), _sum);
                        pA += 16;
                        pK0++;
                    }

                    _sum = _mm512_mul_ps(_sum, _scale);
                    if (pM)
                    {
                        _sum = _mm512_add_ps(_sum, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pM)));
                        pM += 16;
                    }
                    _max = _mm512_max_ps(_max, _sum);
                    _mm512_storeu_ps(scoreptr, _sum);
                    scoreptr += 16;
                    pK += head_dim;
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
                const unsigned short* value = value_head.row<const unsigned short>(n);
                const unsigned short* valueptr = value;
                int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
                    const unsigned short* pV = valueptr;
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
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
                    const unsigned short* pV = valueptr;
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
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
                    const unsigned short* pV = valueptr;
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m512 _p = _mm512_loadu_ps(pS);
                        __m512 _v = _mm512_broadcast_f32x4(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV)));
                        _out0 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                        _out1 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                        _out2 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                        _out3 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
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
            else
            {
                const float* packed_value_tile = packed_value_head.row(n / block_n);
                const float* pV = packed_value_tile;
                int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
#endif // defined(__x86_64__) || defined(_M_X64)
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
        const unsigned short* queryT = queryT_base + ii * head_dim;
        const Mat key_head = key.channel(g);
        const Mat packed_key_head = packed_key.empty() ? Mat() : packed_key.channel(g);
        const Mat value_head = value.channel(g);
        const Mat packed_value_head = packed_value.empty() ? Mat() : packed_value.channel(g);
        const unsigned short* maskT = packed_mask.empty() ? 0 : (const unsigned short*)packed_mask + (size_t)ii * key_seqlen;

        memset(outT, 0, (size_t)value_dim * 8 * sizeof(float));
        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();
        const unsigned short* pM = maskT ? maskT + (size_t)n_begin * 8 : 0;
        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m256 _block_max;
            float* scoreptr = scoreT;
            if (packed_key.empty())
            {
                const unsigned short* key = key_head.row<const unsigned short>(n);
                const __m256 _scale = _mm256_set1_ps(scale);
                __m256 _max = _mm256_set1_ps(-FLT_MAX);
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
                for (; j + 7 < max_jj; j += 8)
                {
                    const unsigned short* pQ = queryT;
                    const unsigned short* pK0 = key + (size_t)j * head_dim;
                    const unsigned short* pK1 = pK0 + head_dim;
                    const unsigned short* pK2 = pK1 + head_dim;
                    const unsigned short* pK3 = pK2 + head_dim;
                    const unsigned short* pK4 = pK3 + head_dim;
                    const unsigned short* pK5 = pK4 + head_dim;
                    const unsigned short* pK6 = pK5 + head_dim;
                    const unsigned short* pK7 = pK6 + head_dim;
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _q = _mm256_loadu_si256((const __m256i*)pQ);
                        _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK0)[0]));
                        _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK1)[0]));
                        _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK2)[0]));
                        _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK3)[0]));
                        _sum4 = _mm256_dpbf16_ps(_sum4, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK4)[0]));
                        _sum5 = _mm256_dpbf16_ps(_sum5, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK5)[0]));
                        _sum6 = _mm256_dpbf16_ps(_sum6, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK6)[0]));
                        _sum7 = _mm256_dpbf16_ps(_sum7, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK7)[0]));
                        pQ += 16;
                        pK0 += 2;
                        pK1 += 2;
                        pK2 += 2;
                        pK3 += 2;
                        pK4 += 2;
                        pK5 += 2;
                        pK6 += 2;
                        pK7 += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m256 _q = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pQ));
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK0++)), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK1++)), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK2++)), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK3++)), _sum3);
                        _sum4 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK4++)), _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK5++)), _sum5);
                        _sum6 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK6++)), _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK7++)), _sum7);
                        pQ += 8;
                    }
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
                        _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)pM)));
                        _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 8))));
                        _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 16))));
                        _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 24))));
                        _sum4 = _mm256_add_ps(_sum4, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 32))));
                        _sum5 = _mm256_add_ps(_sum5, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 40))));
                        _sum6 = _mm256_add_ps(_sum6, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 48))));
                        _sum7 = _mm256_add_ps(_sum7, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 56))));
                        pM += 64;
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
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j + 3 < max_jj; j += 4)
                {
                    const unsigned short* pQ = queryT;
                    const unsigned short* pK0 = key + (size_t)j * head_dim;
                    const unsigned short* pK1 = pK0 + head_dim;
                    const unsigned short* pK2 = pK1 + head_dim;
                    const unsigned short* pK3 = pK2 + head_dim;
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _q = _mm256_loadu_si256((const __m256i*)pQ);
                        _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK0)[0]));
                        _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK1)[0]));
                        _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK2)[0]));
                        _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK3)[0]));
                        pQ += 16;
                        pK0 += 2;
                        pK1 += 2;
                        pK2 += 2;
                        pK3 += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m256 _q = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pQ));
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK0++)), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK1++)), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK2++)), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(*pK3++)), _sum3);
                        pQ += 8;
                    }
                    _sum0 = _mm256_mul_ps(_sum0, _scale);
                    _sum1 = _mm256_mul_ps(_sum1, _scale);
                    _sum2 = _mm256_mul_ps(_sum2, _scale);
                    _sum3 = _mm256_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)pM)));
                        _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 8))));
                        _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 16))));
                        _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 24))));
                        pM += 32;
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
                    const unsigned short* pQ = queryT;
                    const unsigned short* pK = key + (size_t)j * head_dim;
                    __m256 _sum = _mm256_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _q = _mm256_loadu_si256((const __m256i*)pQ);
                        _sum = _mm256_dpbf16_ps(_sum, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK)[0]));
                        pQ += 16;
                        pK += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pQ)), _mm256_set1_ps(bfloat16_to_float32(*pK++)), _sum);
                        pQ += 8;
                    }
                    _sum = _mm256_mul_ps(_sum, _scale);
                    if (pM)
                    {
                        _sum = _mm256_add_ps(_sum, bfloat2float_avx(_mm_loadu_si128((const __m128i*)pM)));
                        pM += 8;
                    }
                    _max = _mm256_max_ps(_max, _sum);
                    _mm256_storeu_ps(scoreptr, _sum);
                    scoreptr += 8;
                }

                _block_max = _max;
            }
            else
            {
                const unsigned short* pK = packed_key_head.row<const unsigned short>(n / block_n);
                const unsigned short* pQ = queryT;
                const __m256 _scale = _mm256_set1_ps(scale);
                __m256 _max = _mm256_set1_ps(-FLT_MAX);
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
                for (; j + 15 < max_jj; j += 16)
                {
                    __m512 _sum0x = _mm512_setzero_ps();
                    __m512 _sum1x = _mm512_setzero_ps();
                    __m512 _sum2x = _mm512_setzero_ps();
                    __m512 _sum3x = _mm512_setzero_ps();
                    __m512 _sum4x = _mm512_setzero_ps();
                    __m512 _sum5x = _mm512_setzero_ps();
                    __m512 _sum6x = _mm512_setzero_ps();
                    __m512 _sum7x = _mm512_setzero_ps();

                    const unsigned short* pA = pQ;
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                        __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pK);
                        __m512i _pA00 = combine8x2_epi32(_pA0, _pA0);
                        __m512i _pA11 = _mm512_shuffle_epi32(_pA00, _MM_PERM_BADC);
                        __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                        __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
                        _sum0x = _mm512_dpbf16_ps(_sum0x, (__m512bh)_pA00, (__m512bh)_pB0);
                        _sum1x = _mm512_dpbf16_ps(_sum1x, (__m512bh)_pA00, (__m512bh)_pB1);
                        _sum2x = _mm512_dpbf16_ps(_sum2x, (__m512bh)_pA11, (__m512bh)_pB0);
                        _sum3x = _mm512_dpbf16_ps(_sum3x, (__m512bh)_pA11, (__m512bh)_pB1);
                        _sum4x = _mm512_dpbf16_ps(_sum4x, (__m512bh)_pA00, (__m512bh)_pB2);
                        _sum5x = _mm512_dpbf16_ps(_sum5x, (__m512bh)_pA00, (__m512bh)_pB3);
                        _sum6x = _mm512_dpbf16_ps(_sum6x, (__m512bh)_pA11, (__m512bh)_pB2);
                        _sum7x = _mm512_dpbf16_ps(_sum7x, (__m512bh)_pA11, (__m512bh)_pB3);
                        pA += 16;
                        pK += 32;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m256 _pAA = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                        __m512 _pA0 = _mm512_castsi512_ps(combine8x2_epi32(_mm256_castps_si256(_pAA), _mm256_castps_si256(_pAA)));
                        __m512 _pB0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pK));

                        __m512 _pA1 = _mm512_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        __m512 _pB2 = _mm512_castsi512_ps(_mm512_permutex_epi64(_mm512_castps_si512(_pB0), _MM_SHUFFLE(1, 0, 3, 2)));
                        __m512 _pB3 = _mm512_permute_ps(_pB2, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0x = _mm512_fmadd_ps(_pA0, _pB0, _sum0x);
                        _sum1x = _mm512_fmadd_ps(_pA0, _pB1, _sum1x);
                        _sum2x = _mm512_fmadd_ps(_pA1, _pB0, _sum2x);
                        _sum3x = _mm512_fmadd_ps(_pA1, _pB1, _sum3x);
                        _sum4x = _mm512_fmadd_ps(_pA0, _pB2, _sum4x);
                        _sum5x = _mm512_fmadd_ps(_pA0, _pB3, _sum5x);
                        _sum6x = _mm512_fmadd_ps(_pA1, _pB2, _sum6x);
                        _sum7x = _mm512_fmadd_ps(_pA1, _pB3, _sum7x);
                        pA += 8;
                        pK += 16;
                    }

                    _sum1x = _mm512_permute_ps(_sum1x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum3x = _mm512_permute_ps(_sum3x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum5x = _mm512_permute_ps(_sum5x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum7x = _mm512_permute_ps(_sum7x, _MM_SHUFFLE(2, 1, 0, 3));

                    __m512 _tmp0 = _mm512_unpacklo_ps(_sum0x, _sum3x);
                    __m512 _tmp1 = _mm512_unpackhi_ps(_sum0x, _sum3x);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_sum2x, _sum1x);
                    __m512 _tmp3 = _mm512_unpackhi_ps(_sum2x, _sum1x);
                    __m512 _tmp4 = _mm512_unpacklo_ps(_sum4x, _sum7x);
                    __m512 _tmp5 = _mm512_unpackhi_ps(_sum4x, _sum7x);
                    __m512 _tmp6 = _mm512_unpacklo_ps(_sum6x, _sum5x);
                    __m512 _tmp7 = _mm512_unpackhi_ps(_sum6x, _sum5x);

                    _sum0x = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                    _sum1x = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                    _sum2x = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                    _sum3x = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                    _sum4x = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                    _sum5x = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                    _sum6x = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                    _sum7x = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));

                    _sum1x = _mm512_permute_ps(_sum1x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum3x = _mm512_permute_ps(_sum3x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum5x = _mm512_permute_ps(_sum5x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum7x = _mm512_permute_ps(_sum7x, _MM_SHUFFLE(2, 1, 0, 3));

                    _tmp0 = _mm512_shuffle_f32x4(_sum0x, _sum4x, _MM_SHUFFLE(0, 1, 1, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_sum0x, _sum4x, _MM_SHUFFLE(2, 3, 3, 2));
                    _tmp2 = _mm512_shuffle_f32x4(_sum1x, _sum5x, _MM_SHUFFLE(0, 1, 1, 0));
                    _tmp3 = _mm512_shuffle_f32x4(_sum1x, _sum5x, _MM_SHUFFLE(2, 3, 3, 2));
                    _tmp4 = _mm512_shuffle_f32x4(_sum2x, _sum6x, _MM_SHUFFLE(0, 1, 1, 0));
                    _tmp5 = _mm512_shuffle_f32x4(_sum2x, _sum6x, _MM_SHUFFLE(2, 3, 3, 2));
                    _tmp6 = _mm512_shuffle_f32x4(_sum3x, _sum7x, _MM_SHUFFLE(0, 1, 1, 0));
                    _tmp7 = _mm512_shuffle_f32x4(_sum3x, _sum7x, _MM_SHUFFLE(2, 3, 3, 2));

                    _sum0x = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1x = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2x = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3x = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum4x = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum5x = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum6x = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum7x = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));

                    __m256 _sum0 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum0x, 0), _scale);
                    __m256 _sum1 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum1x, 0), _scale);
                    __m256 _sum2 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum2x, 0), _scale);
                    __m256 _sum3 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum3x, 0), _scale);
                    __m256 _sum4 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum4x, 0), _scale);
                    __m256 _sum5 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum5x, 0), _scale);
                    __m256 _sum6 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum6x, 0), _scale);
                    __m256 _sum7 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum7x, 0), _scale);
                    __m256 _sum8 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum0x, 1), _scale);
                    __m256 _sum9 = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum1x, 1), _scale);
                    __m256 _suma = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum2x, 1), _scale);
                    __m256 _sumb = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum3x, 1), _scale);
                    __m256 _sumc = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum4x, 1), _scale);
                    __m256 _sumd = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum5x, 1), _scale);
                    __m256 _sume = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum6x, 1), _scale);
                    __m256 _sumf = _mm256_mul_ps(_mm512_extractf32x8_ps(_sum7x, 1), _scale);
                    if (pM)
                    {
                        _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)pM)));
                        _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 8))));
                        _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 16))));
                        _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 24))));
                        _sum4 = _mm256_add_ps(_sum4, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 32))));
                        _sum5 = _mm256_add_ps(_sum5, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 40))));
                        _sum6 = _mm256_add_ps(_sum6, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 48))));
                        _sum7 = _mm256_add_ps(_sum7, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 56))));
                        _sum8 = _mm256_add_ps(_sum8, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 64))));
                        _sum9 = _mm256_add_ps(_sum9, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 72))));
                        _suma = _mm256_add_ps(_suma, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 80))));
                        _sumb = _mm256_add_ps(_sumb, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 88))));
                        _sumc = _mm256_add_ps(_sumc, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 96))));
                        _sumd = _mm256_add_ps(_sumd, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 104))));
                        _sume = _mm256_add_ps(_sume, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 112))));
                        _sumf = _mm256_add_ps(_sumf, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 120))));
                        pM += 128;
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
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                        __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pK);
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
                        pK += 16;
                    }
#endif // __AVX512BF16__
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
                        pK += 8;
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
                        _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)pM)));
                        _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 8))));
                        _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 16))));
                        _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 24))));
                        _sum4 = _mm256_add_ps(_sum4, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 32))));
                        _sum5 = _mm256_add_ps(_sum5, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 40))));
                        _sum6 = _mm256_add_ps(_sum6, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 48))));
                        _sum7 = _mm256_add_ps(_sum7, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 56))));
                        pM += 64;
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
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j + 3 < max_jj; j += 4)
                {
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();

                    const unsigned short* pA = pQ;
                    int d = 0;
#if __AVX512BF16__
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
                        pK += 8;
                    }
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
                        pK += 4;
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

                    _sum0 = _mm256_mul_ps(_sum0, _scale);
                    _sum1 = _mm256_mul_ps(_sum1, _scale);
                    _sum2 = _mm256_mul_ps(_sum2, _scale);
                    _sum3 = _mm256_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)pM)));
                        _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 8))));
                        _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 16))));
                        _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + 24))));
                        pM += 32;
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
                    const unsigned short* pA = pQ;
                    const unsigned short* pK0 = pK;
                    __m256 _sum = _mm256_setzero_ps();

                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                        _sum = _mm256_dpbf16_ps(_sum, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pK0)[0]));
                        pA += 16;
                        pK0 += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m256 _pA = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                        _sum = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pK0[0])), _sum);
                        pA += 8;
                        pK0++;
                    }

                    _sum = _mm256_mul_ps(_sum, _scale);
                    if (pM)
                    {
                        _sum = _mm256_add_ps(_sum, bfloat2float_avx(_mm_loadu_si128((const __m128i*)pM)));
                        pM += 8;
                    }
                    _max = _mm256_max_ps(_max, _sum);
                    _mm256_storeu_ps(scoreptr, _sum);
                    scoreptr += 8;
                    pK += head_dim;
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
                const unsigned short* value = value_head.row<const unsigned short>(n);
                const unsigned short* valueptr = value;
                int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
                    const unsigned short* pV = valueptr;
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m256 _p = _mm256_loadu_ps(pS);
                        __m512 _v = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV));
                        __m256 _v0 = _mm512_extractf32x8_ps(_v, 0);
                        __m256 _v00 = _mm256_permute2f128_ps(_v0, _v0, 0x00);
                        _out0 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v00, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                        _out1 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v00, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                        _out2 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v00, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                        _out3 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v00, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                        __m256 _v01 = _mm256_permute2f128_ps(_v0, _v0, 0x11);
                        _out4 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v01, _MM_SHUFFLE(0, 0, 0, 0)), _out4);
                        _out5 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v01, _MM_SHUFFLE(1, 1, 1, 1)), _out5);
                        _out6 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v01, _MM_SHUFFLE(2, 2, 2, 2)), _out6);
                        _out7 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v01, _MM_SHUFFLE(3, 3, 3, 3)), _out7);
                        __m256 _v1 = _mm512_extractf32x8_ps(_v, 1);
                        __m256 _v10 = _mm256_permute2f128_ps(_v1, _v1, 0x00);
                        _out8 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v10, _MM_SHUFFLE(0, 0, 0, 0)), _out8);
                        _out9 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v10, _MM_SHUFFLE(1, 1, 1, 1)), _out9);
                        _outa = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v10, _MM_SHUFFLE(2, 2, 2, 2)), _outa);
                        _outb = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v10, _MM_SHUFFLE(3, 3, 3, 3)), _outb);
                        __m256 _v11 = _mm256_permute2f128_ps(_v1, _v1, 0x11);
                        _outc = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v11, _MM_SHUFFLE(0, 0, 0, 0)), _outc);
                        _outd = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v11, _MM_SHUFFLE(1, 1, 1, 1)), _outd);
                        _oute = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v11, _MM_SHUFFLE(2, 2, 2, 2)), _oute);
                        _outf = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v11, _MM_SHUFFLE(3, 3, 3, 3)), _outf);
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
                    const unsigned short* pV = valueptr;
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
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
                    const unsigned short* pV = valueptr;
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m256 _p = _mm256_loadu_ps(pS);
                        __m128 _v = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV));
                        __m256 _v0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_v), _v, 1);
                        _out0 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v0, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                        _out1 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v0, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                        _out2 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v0, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                        _out3 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v0, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
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
            else
            {
                const float* packed_value_tile = packed_value_head.row(n / block_n);
                const float* pV = packed_value_tile;
                int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
#endif // defined(__x86_64__) || defined(_M_X64)
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
        const unsigned short* queryT = queryT_base + ii * head_dim;
        const Mat key_head = key.channel(g);
        const Mat packed_key_head = packed_key.empty() ? Mat() : packed_key.channel(g);
        const Mat value_head = value.channel(g);
        const Mat packed_value_head = packed_value.empty() ? Mat() : packed_value.channel(g);
        const unsigned short* maskT = packed_mask.empty() ? 0 : (const unsigned short*)packed_mask + (size_t)ii * key_seqlen;

        memset(outT, 0, (size_t)value_dim * 4 * sizeof(float));
        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();
        const unsigned short* pM = maskT ? maskT + (size_t)n_begin * 4 : 0;

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m128 _block_max;
            float* scoreptr = scoreT;
            if (packed_key.empty())
            {
                const unsigned short* key = key_head.row<const unsigned short>(n);
                const __m128 _scale = _mm_set1_ps(scale);
                __m128 _max = _mm_set1_ps(-FLT_MAX);
                int j = 0;
                for (; j + 3 < max_jj; j += 4)
                {
                    const unsigned short* pQ = queryT;
                    const unsigned short* pK0 = key + (size_t)j * head_dim;
                    const unsigned short* pK1 = pK0 + head_dim;
                    const unsigned short* pK2 = pK1 + head_dim;
                    const unsigned short* pK3 = pK2 + head_dim;
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ));
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK0++)), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK1++)), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK2++)), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK3++)), _sum3);
                        pQ += 4;
                    }
                    _sum0 = _mm_mul_ps(_sum0, _scale);
                    _sum1 = _mm_mul_ps(_sum1, _scale);
                    _sum2 = _mm_mul_ps(_sum2, _scale);
                    _sum3 = _mm_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pM)));
                        _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 4))));
                        _sum2 = _mm_add_ps(_sum2, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 8))));
                        _sum3 = _mm_add_ps(_sum3, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 12))));
                        pM += 16;
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
                    const unsigned short* pQ = queryT;
                    const unsigned short* pK = key + (size_t)j * head_dim;
                    __m128 _sum = _mm_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q = _mm_loadu_si128((const __m128i*)pQ);
                        _sum = _mm_dpbf16_ps(_sum, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK)[0]));
                        pQ += 8;
                        pK += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ)), _mm_set1_ps(bfloat16_to_float32(*pK++)), _sum);
                        pQ += 4;
                    }
                    _sum = _mm_mul_ps(_sum, _scale);
                    if (pM)
                    {
                        _sum = _mm_add_ps(_sum, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pM)));
                        pM += 4;
                    }
                    _max = _mm_max_ps(_max, _sum);
                    _mm_storeu_ps(scoreptr, _sum);
                    scoreptr += 4;
                }

                _block_max = _max;
            }
            else
            {
                const unsigned short* pK = packed_key_head.row<const unsigned short>(n / block_n);
                const unsigned short* pQ = queryT;
                const __m128 _scale = _mm_set1_ps(scale);
                __m128 _max = _mm_set1_ps(-FLT_MAX);
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
                for (; j + 15 < max_jj; j += 16)
                {
                    __m512 _sum0x = _mm512_setzero_ps();
                    __m512 _sum1x = _mm512_setzero_ps();
                    __m512 _sum2x = _mm512_setzero_ps();
                    __m512 _sum3x = _mm512_setzero_ps();
                    const unsigned short* pA = pQ;
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _pA0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pA));
                        __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pK);
                        __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                        __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                        _sum0x = _mm512_dpbf16_ps(_sum0x, (__m512bh)_pA0, (__m512bh)_pB0);
                        _sum1x = _mm512_dpbf16_ps(_sum1x, (__m512bh)_pA0, (__m512bh)_pB1);
                        _sum2x = _mm512_dpbf16_ps(_sum2x, (__m512bh)_pA1, (__m512bh)_pB0);
                        _sum3x = _mm512_dpbf16_ps(_sum3x, (__m512bh)_pA1, (__m512bh)_pB1);
                        pA += 8;
                        pK += 32;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m128 _pAs = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                        __m512 _pA0 = _mm512_broadcast_f32x4(_pAs);
                        __m512 _pB0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pK));

                        __m512 _pA1 = _mm512_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0x = _mm512_fmadd_ps(_pA0, _pB0, _sum0x);
                        _sum1x = _mm512_fmadd_ps(_pA0, _pB1, _sum1x);
                        _sum2x = _mm512_fmadd_ps(_pA1, _pB0, _sum2x);
                        _sum3x = _mm512_fmadd_ps(_pA1, _pB1, _sum3x);
                        pA += 4;
                        pK += 16;
                    }

                    _sum1x = _mm512_permute_ps(_sum1x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum3x = _mm512_permute_ps(_sum3x, _MM_SHUFFLE(2, 1, 0, 3));
                    __m512 _tmp0 = _mm512_unpacklo_ps(_sum0x, _sum3x);
                    __m512 _tmp1 = _mm512_unpacklo_ps(_sum2x, _sum1x);
                    __m512 _tmp2 = _mm512_unpackhi_ps(_sum0x, _sum3x);
                    __m512 _tmp3 = _mm512_unpackhi_ps(_sum2x, _sum1x);
                    _sum0x = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _sum1x = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _sum2x = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                    _sum3x = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                    _sum1x = _mm512_permute_ps(_sum1x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum3x = _mm512_permute_ps(_sum3x, _MM_SHUFFLE(2, 1, 0, 3));

                    _tmp0 = _mm512_shuffle_f32x4(_sum0x, _sum1x, _MM_SHUFFLE(1, 0, 1, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_sum2x, _sum3x, _MM_SHUFFLE(1, 0, 1, 0));
                    _tmp2 = _mm512_shuffle_f32x4(_sum0x, _sum1x, _MM_SHUFFLE(3, 2, 3, 2));
                    _tmp3 = _mm512_shuffle_f32x4(_sum2x, _sum3x, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum0x = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1x = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2x = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3x = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    __m128 _sum0 = _mm512_extractf32x4_ps(_sum0x, 0);
                    __m128 _sum1 = _mm512_extractf32x4_ps(_sum0x, 1);
                    __m128 _sum2 = _mm512_extractf32x4_ps(_sum0x, 2);
                    __m128 _sum3 = _mm512_extractf32x4_ps(_sum0x, 3);
                    __m128 _sum4 = _mm512_extractf32x4_ps(_sum1x, 0);
                    __m128 _sum5 = _mm512_extractf32x4_ps(_sum1x, 1);
                    __m128 _sum6 = _mm512_extractf32x4_ps(_sum1x, 2);
                    __m128 _sum7 = _mm512_extractf32x4_ps(_sum1x, 3);
                    __m128 _sum8 = _mm512_extractf32x4_ps(_sum2x, 0);
                    __m128 _sum9 = _mm512_extractf32x4_ps(_sum2x, 1);
                    __m128 _suma = _mm512_extractf32x4_ps(_sum2x, 2);
                    __m128 _sumb = _mm512_extractf32x4_ps(_sum2x, 3);
                    __m128 _sumc = _mm512_extractf32x4_ps(_sum3x, 0);
                    __m128 _sumd = _mm512_extractf32x4_ps(_sum3x, 1);
                    __m128 _sume = _mm512_extractf32x4_ps(_sum3x, 2);
                    __m128 _sumf = _mm512_extractf32x4_ps(_sum3x, 3);

                    _sum0 = _mm_mul_ps(_sum0, _scale);
                    _sum1 = _mm_mul_ps(_sum1, _scale);
                    _sum2 = _mm_mul_ps(_sum2, _scale);
                    _sum3 = _mm_mul_ps(_sum3, _scale);
                    _sum4 = _mm_mul_ps(_sum4, _scale);
                    _sum5 = _mm_mul_ps(_sum5, _scale);
                    _sum6 = _mm_mul_ps(_sum6, _scale);
                    _sum7 = _mm_mul_ps(_sum7, _scale);
                    _sum8 = _mm_mul_ps(_sum8, _scale);
                    _sum9 = _mm_mul_ps(_sum9, _scale);
                    _suma = _mm_mul_ps(_suma, _scale);
                    _sumb = _mm_mul_ps(_sumb, _scale);
                    _sumc = _mm_mul_ps(_sumc, _scale);
                    _sumd = _mm_mul_ps(_sumd, _scale);
                    _sume = _mm_mul_ps(_sume, _scale);
                    _sumf = _mm_mul_ps(_sumf, _scale);
                    if (pM)
                    {
                        _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pM)));
                        _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 4))));
                        _sum2 = _mm_add_ps(_sum2, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 8))));
                        _sum3 = _mm_add_ps(_sum3, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 12))));
                        _sum4 = _mm_add_ps(_sum4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 16))));
                        _sum5 = _mm_add_ps(_sum5, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 20))));
                        _sum6 = _mm_add_ps(_sum6, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 24))));
                        _sum7 = _mm_add_ps(_sum7, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 28))));
                        _sum8 = _mm_add_ps(_sum8, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 32))));
                        _sum9 = _mm_add_ps(_sum9, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 36))));
                        _suma = _mm_add_ps(_suma, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 40))));
                        _sumb = _mm_add_ps(_sumb, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 44))));
                        _sumc = _mm_add_ps(_sumc, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 48))));
                        _sumd = _mm_add_ps(_sumd, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 52))));
                        _sume = _mm_add_ps(_sume, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 56))));
                        _sumf = _mm_add_ps(_sumf, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 60))));
                        pM += 64;
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
                    __m256 _sum0x = _mm256_setzero_ps();
                    __m256 _sum1x = _mm256_setzero_ps();
                    __m256 _sum2x = _mm256_setzero_ps();
                    __m256 _sum3x = _mm256_setzero_ps();
                    const unsigned short* pA = pQ;
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                        __m256i _pA00 = combine4x2_epi32(_pA0, _pA0);
                        __m256i _pB01 = _mm256_loadu_si256((const __m256i*)pK);
                        __m256i _pA11 = _mm256_shuffle_epi32(_pA00, _MM_SHUFFLE(1, 0, 3, 2));
                        __m256i _pB23 = _mm256_shuffle_epi32(_pB01, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0x = _mm256_dpbf16_ps(_sum0x, (__m256bh)_pA00, (__m256bh)_pB01);
                        _sum1x = _mm256_dpbf16_ps(_sum1x, (__m256bh)_pA11, (__m256bh)_pB01);
                        _sum2x = _mm256_dpbf16_ps(_sum2x, (__m256bh)_pA00, (__m256bh)_pB23);
                        _sum3x = _mm256_dpbf16_ps(_sum3x, (__m256bh)_pA11, (__m256bh)_pB23);
                        pA += 8;
                        pK += 16;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m128 _pA = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                        __m256 _pA0 = combine4x2_ps(_pA, _pA);
                        __m256 _pB0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pK));

                        __m256 _pA1 = _mm256_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m256 _pB1 = _mm256_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0x = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0x);
                        _sum1x = _mm256_comp_fmadd_ps(_pA1, _pB0, _sum1x);
                        _sum2x = _mm256_comp_fmadd_ps(_pA0, _pB1, _sum2x);
                        _sum3x = _mm256_comp_fmadd_ps(_pA1, _pB1, _sum3x);
                        pA += 4;
                        pK += 8;
                    }

                    _sum2x = _mm256_shuffle_ps(_sum2x, _sum2x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum3x = _mm256_shuffle_ps(_sum3x, _sum3x, _MM_SHUFFLE(2, 1, 0, 3));
                    __m256 _tmp0 = _mm256_unpacklo_ps(_sum0x, _sum3x);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_sum0x, _sum3x);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_sum1x, _sum2x);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_sum1x, _sum2x);
                    _sum0x = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                    _sum1x = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                    _sum2x = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                    _sum3x = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                    _sum1x = _mm256_shuffle_ps(_sum1x, _sum1x, _MM_SHUFFLE(2, 1, 0, 3));
                    _sum3x = _mm256_shuffle_ps(_sum3x, _sum3x, _MM_SHUFFLE(2, 1, 0, 3));

                    __m128 _sum0 = _mm_mul_ps(_mm256_extractf128_ps(_sum0x, 0), _scale);
                    __m128 _sum1 = _mm_mul_ps(_mm256_extractf128_ps(_sum1x, 0), _scale);
                    __m128 _sum2 = _mm_mul_ps(_mm256_extractf128_ps(_sum2x, 0), _scale);
                    __m128 _sum3 = _mm_mul_ps(_mm256_extractf128_ps(_sum3x, 0), _scale);
                    __m128 _sum4 = _mm_mul_ps(_mm256_extractf128_ps(_sum0x, 1), _scale);
                    __m128 _sum5 = _mm_mul_ps(_mm256_extractf128_ps(_sum1x, 1), _scale);
                    __m128 _sum6 = _mm_mul_ps(_mm256_extractf128_ps(_sum2x, 1), _scale);
                    __m128 _sum7 = _mm_mul_ps(_mm256_extractf128_ps(_sum3x, 1), _scale);
                    if (pM)
                    {
                        _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pM)));
                        _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 4))));
                        _sum2 = _mm_add_ps(_sum2, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 8))));
                        _sum3 = _mm_add_ps(_sum3, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 12))));
                        _sum4 = _mm_add_ps(_sum4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 16))));
                        _sum5 = _mm_add_ps(_sum5, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 20))));
                        _sum6 = _mm_add_ps(_sum6, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 24))));
                        _sum7 = _mm_add_ps(_sum7, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 28))));
                        pM += 32;
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
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j + 3 < max_jj; j += 4)
                {
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();

                    const unsigned short* pA = pQ;
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                        __m128i _pB0 = _mm_loadu_si128((const __m128i*)pK);
                        __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA0, (__m128bh)_pB0);
                        _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA0, (__m128bh)_pB1);
                        _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_pA1, (__m128bh)_pB0);
                        _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_pA1, (__m128bh)_pB1);
                        pA += 8;
                        pK += 8;
                    }
#endif // __AVX512BF16__
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
                        pK += 4;
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

                    _sum0 = _mm_mul_ps(_sum0, _scale);
                    _sum1 = _mm_mul_ps(_sum1, _scale);
                    _sum2 = _mm_mul_ps(_sum2, _scale);
                    _sum3 = _mm_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pM)));
                        _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 4))));
                        _sum2 = _mm_add_ps(_sum2, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 8))));
                        _sum3 = _mm_add_ps(_sum3, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + 12))));
                        pM += 16;
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
                    const unsigned short* pA = pQ;
                    const unsigned short* pK0 = pK;
                    __m128 _sum = _mm_setzero_ps();

                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                        _sum = _mm_dpbf16_ps(_sum, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pK0)[0]));
                        pA += 8;
                        pK0 += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m128 _pA = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                        _sum = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pK0[0])), _sum);
                        pA += 4;
                        pK0++;
                    }

                    _sum = _mm_mul_ps(_sum, _scale);
                    if (pM)
                    {
                        _sum = _mm_add_ps(_sum, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pM)));
                        pM += 4;
                    }
                    _max = _mm_max_ps(_max, _sum);
                    _mm_storeu_ps(scoreptr, _sum);
                    scoreptr += 4;
                    pK += head_dim;
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
            else
            {
                const float* packed_value_tile = packed_value_head.row(n / block_n);
                const float* pV = packed_value_tile;
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
#endif // defined(__x86_64__) || defined(_M_X64)
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
        const unsigned short* mask = mask_head.empty() ? 0 : mask_head.row<const unsigned short>(i0x);
        const unsigned short* qptr = query_head.row<const unsigned short>(i0x);

        memset(out, 0, value_dim * sizeof(float));
        float m = -FLT_MAX;
        float l = 0.f;
        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            float block_max = -FLT_MAX;
            for (int j = 0; j < max_jj; j++)
            {
                const unsigned short* kptr = key_head.row<const unsigned short>(n + j);
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
                float sum = 0.f;

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
#if __AVX512BF16__
                for (; i + 31 < head_dim; i += 32)
                {
                    __m512i _q = _mm512_loadu_si512((const __m512i*)(qptr + i));
                    __m512i _k = _mm512_loadu_si512((const __m512i*)(kptr + i));
                    _sum_bf16 = _mm512_dpbf16_ps(_sum_bf16, (__m512bh)_q, (__m512bh)_k);
                }
#endif // __AVX512BF16__
                for (; i + 15 < head_dim; i += 16)
                {
                    __m512 _q = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr + i)));
                    __m512 _k = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + i)));
                    _sum_avx512 = _mm512_fmadd_ps(_q, _k, _sum_avx512);
                }
#endif // __AVX512F__
                for (; i + 7 < head_dim; i += 8)
                {
                    __m256 _q = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr + i)));
                    __m256 _k = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(kptr + i)));
                    _sum_avx = _mm256_comp_fmadd_ps(_q, _k, _sum_avx);
                }
#endif // __AVX__
                for (; i + 3 < head_dim; i += 4)
                {
                    __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(qptr + i)));
                    __m128 _k = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(kptr + i)));
                    _sum = _mm_comp_fmadd_ps(_q, _k, _sum);
                }
#endif // __SSE2__
                for (; i < head_dim; i++)
                    sum += bfloat16_to_float32(qptr[i]) * bfloat16_to_float32(kptr[i]);

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
                if (mask)
                    s += bfloat16_to_float32(mask[n + j]);
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
                const unsigned short* value = value_head.row<const unsigned short>(n);
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
                        const unsigned short* vptr = value + (size_t)j * value_dim + d;
                        __m512 _p = _mm512_set1_ps(score[j]);
                        _out0 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)vptr)), _p, _out0);
                        _out1 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 16))), _p, _out1);
                        _out2 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 32))), _p, _out2);
                        _out3 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 48))), _p, _out3);
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
                    {
                        const unsigned short* vptr = value + (size_t)j * value_dim + d;
                        _out = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)vptr)), _mm512_set1_ps(score[j]), _out);
                    }
                    _mm512_storeu_ps(out + d, _out);
                }
#else
                for (; d + 15 < value_dim; d += 16)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(out + d + 8), _mm256_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        const unsigned short* vptr = value + (size_t)j * value_dim + d;
                        __m256 _p = _mm256_set1_ps(score[j]);
                        _out0 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)vptr)), _p, _out0);
                        _out1 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(vptr + 8))), _p, _out1);
                    }
                    _mm256_storeu_ps(out + d, _out0);
                    _mm256_storeu_ps(out + d + 8, _out1);
                }
#endif // __AVX512F__

                for (; d + 7 < value_dim; d += 8)
                {
                    __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        const unsigned short* vptr = value + (size_t)j * value_dim + d;
                        _out = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)vptr)), _mm256_set1_ps(score[j]), _out);
                    }
                    _mm256_storeu_ps(out + d, _out);
                }
#else
                for (; d + 7 < value_dim; d += 8)
                {
                    __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
                    __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(out + d + 4), _mm_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        const unsigned short* vptr = value + (size_t)j * value_dim + d;
                        __m128 _p = _mm_set1_ps(score[j]);
                        _out0 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)vptr)), _p, _out0);
                        _out1 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(vptr + 4))), _p, _out1);
                    }
                    _mm_storeu_ps(out + d, _out0);
                    _mm_storeu_ps(out + d + 4, _out1);
                }
#endif // __AVX__
                for (; d + 3 < value_dim; d += 4)
                {
                    __m128 _out = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        const unsigned short* vptr = value + (size_t)j * value_dim + d;
                        _out = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)vptr)), _mm_set1_ps(score[j]), _out);
                    }
                    _mm_storeu_ps(out + d, _out);
                }
#endif // __SSE2__
                for (; d < value_dim; d++)
                {
                    float sum = out[d] * alpha;
                    for (int j = 0; j < max_jj; j++)
                        sum += score[j] * bfloat16_to_float32(value[(size_t)j * value_dim + d]);
                    out[d] = sum;
                }
            }
            else
            {
                const float* packed_value_tile = packed_value_head.row(n / block_n);
                const float* pV = packed_value_tile;
                int d = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
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
#endif // defined(__x86_64__) || defined(_M_X64)
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

    const int query_seqlen = query.h;
    const int num_query_heads = query.c;
    const int num_kv_heads = key.c;
    const int key_seqlen = key.h;
    const int value_dim = value.w;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int nT = std::max(opt.num_threads, 1);
    const int block_m = sdpa_prefill_get_optimal_tile_m(query_seqlen, num_query_heads, nT);
    const int num_mblocks = (query_seqlen + block_m - 1) / block_m;
    const int num_tasks = num_query_heads * num_mblocks;
    const int num_mask_heads = attn_mask_blob.dims == 3 ? attn_mask_blob.c : 1;
    const bool use_packed_mask = !attn_mask_blob.empty() && block_m >= 4;
    const int key_reuse = (query_seqlen + block_m - 1) / block_m * num_query_heads_per_kv_head;
    const bool use_packed_key = query_seqlen >= 4 && key_reuse >= 4;
    int value_pack_reuse = 4;
#if __SSE2__
#if __AVX__
    value_pack_reuse = 3;
#endif // __AVX__
#endif // __SSE2__
    if (value_dim < 32)
        value_pack_reuse += 2;
    const bool use_packed_value = key_reuse >= value_pack_reuse;
    const int block_n = sdpa_prefill_get_optimal_tile_n(query.w, value_dim, key_seqlen, 2, 2, use_packed_value ? 4 : 2, attn_mask_blob.empty() ? 0 : 2, block_m, num_tasks, nT);
    const int state_stride = block_m;

    const int num_key_blocks = (key_seqlen + block_n - 1) / block_n;

    Mat packed_key;
    if (use_packed_key)
    {
        packed_key.create(key.w * block_n, num_key_blocks, num_kv_heads, 2u, opt.workspace_allocator);
        if (packed_key.empty())
            return -100;

        sdpa_pack_key_bf16s(key, packed_key, block_n, opt);
    }

    Mat packed_value;
    if (use_packed_value)
    {
        packed_value.create(value_dim * block_n, num_key_blocks, num_kv_heads, 4u, opt.workspace_allocator);
        if (packed_value.empty())
            return -100;

        sdpa_pack_value_bf16s(value, packed_value, block_n, opt);
    }

    Mat packed_mask;
    if (use_packed_mask)
    {
        packed_mask.create(key_seqlen * block_m, num_mblocks, num_mask_heads, 2u, opt.workspace_allocator);
        if (packed_mask.empty())
            return -100;

        sdpa_pack_mask_bf16s(attn_mask_blob, packed_mask, block_m, opt);
    }

    int num_kv_chunks = 1;
    if (num_tasks < nT && num_key_blocks >= 2)
    {
        num_kv_chunks = std::min((nT + num_tasks - 1) / num_tasks, num_key_blocks);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    Mat packed_query;
    if (num_kv_chunks > 1)
    {
        packed_query.create(query.w * block_m, 1, num_tasks, 2u, opt.workspace_allocator);
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
            sdpa_pack_query_bf16s(query_head, queryT, i0, max_ii);
        }
    }

    const int query_workspace_size = num_kv_chunks > 1 ? 0 : (block_m * query.w + 1) / 2;
    const int workspace_size = (block_m * (block_n + value_dim) + query_workspace_size + 15) / 16 * 16;
    Mat workspace(workspace_size, 1, nT, 4u, opt.workspace_allocator);
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
        sdpa_flash_attention_tile_bf16s(query, key, packed_key, value, packed_value, attn_mask_blob, packed_mask_tile, top_blob, scale, q, g, i0, max_ii, n_begin, n_end, block_n, state_stride, packed_query_tile, workspace_tile, state);
    }

    if (num_kv_chunks > 1)
    {
        sdpa_prefill_reduce(partials, top_blob, workspace, num_tasks, num_mblocks, block_m, num_kv_chunks, query_seqlen, value_dim, opt);
    }

    return 0;
}

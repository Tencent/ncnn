// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// packed_key[block][key_panel][head_dim][key_lane] in fp32
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
        Mat packed_key_head = packed_key.channel(g);
        const unsigned short* key_base = (const unsigned short*)key_head + (size_t)n * head_dim;
        float* pp = packed_key_head.row(block_id);

        const int max_jj = std::min(block_n, key_seqlen - n);
        int j = 0;

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
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p1));
                __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p2));
                __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p3));
                __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p4));
                __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p5));
                __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p6));
                __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p7));
                __m512 _r8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p8));
                __m512 _r9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p9));
                __m512 _ra = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pa));
                __m512 _rb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pb));
                __m512 _rc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pc));
                __m512 _rd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pd));
                __m512 _re = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pe));
                __m512 _rf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pf));

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
                pp[0] = bfloat16_to_float32(*p0++);
                pp[1] = bfloat16_to_float32(*p1++);
                pp[2] = bfloat16_to_float32(*p2++);
                pp[3] = bfloat16_to_float32(*p3++);
                pp[4] = bfloat16_to_float32(*p4++);
                pp[5] = bfloat16_to_float32(*p5++);
                pp[6] = bfloat16_to_float32(*p6++);
                pp[7] = bfloat16_to_float32(*p7++);
                pp[8] = bfloat16_to_float32(*p8++);
                pp[9] = bfloat16_to_float32(*p9++);
                pp[10] = bfloat16_to_float32(*pa++);
                pp[11] = bfloat16_to_float32(*pb++);
                pp[12] = bfloat16_to_float32(*pc++);
                pp[13] = bfloat16_to_float32(*pd++);
                pp[14] = bfloat16_to_float32(*pe++);
                pp[15] = bfloat16_to_float32(*pf++);
                pp += 16;
            }
        }
#endif // __AVX512F__
#if __AVX__
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
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p2));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p3));
                __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p4));
                __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p5));
                __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p6));
                __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p7));

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
                pp[0] = bfloat16_to_float32(*p0++);
                pp[1] = bfloat16_to_float32(*p1++);
                pp[2] = bfloat16_to_float32(*p2++);
                pp[3] = bfloat16_to_float32(*p3++);
                pp[4] = bfloat16_to_float32(*p4++);
                pp[5] = bfloat16_to_float32(*p5++);
                pp[6] = bfloat16_to_float32(*p6++);
                pp[7] = bfloat16_to_float32(*p7++);
                pp += 8;
            }
        }
#endif // __AVX__
#if __SSE2__
        for (; j + 3 < max_jj; j += 4)
        {
            const unsigned short* p0 = key_base + (size_t)j * head_dim;
            const unsigned short* p1 = p0 + head_dim;
            const unsigned short* p2 = p1 + head_dim;
            const unsigned short* p3 = p2 + head_dim;

            int k = 0;
            for (; k + 3 < head_dim; k += 4)
            {
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1));
                __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p2));
                __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p3));

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
                pp[0] = bfloat16_to_float32(*p0++);
                pp[1] = bfloat16_to_float32(*p1++);
                pp[2] = bfloat16_to_float32(*p2++);
                pp[3] = bfloat16_to_float32(*p3++);
                pp += 4;
            }
        }
#endif // __SSE2__

        for (; j < max_jj; j++)
        {
            const unsigned short* p0 = key_base + (size_t)j * head_dim;
            for (int d = 0; d < head_dim; d++)
                pp[d] = bfloat16_to_float32(p0[d]);
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
#if __AVX__
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
#if __SSE2__
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

// queryT[head_dim][query_lane] in fp32
static void sdpa_pack_query_bf16s(const Mat& query_head, Mat& queryT, int i, int max_ii, float scale)
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

        const __m512 _scale = _mm512_set1_ps(scale);
        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr0 + d)));
            __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr1 + d)));
            __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr2 + d)));
            __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr3 + d)));
            __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr4 + d)));
            __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr5 + d)));
            __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr6 + d)));
            __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr7 + d)));
            __m512 _r8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr8 + d)));
            __m512 _r9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptr9 + d)));
            __m512 _ra = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptra + d)));
            __m512 _rb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptrb + d)));
            __m512 _rc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptrc + d)));
            __m512 _rd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptrd + d)));
            __m512 _re = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptre + d)));
            __m512 _rf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(qptrf + d)));
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
            pQ[0] = bfloat16_to_float32(qptr0[d]) * scale;
            pQ[1] = bfloat16_to_float32(qptr1[d]) * scale;
            pQ[2] = bfloat16_to_float32(qptr2[d]) * scale;
            pQ[3] = bfloat16_to_float32(qptr3[d]) * scale;
            pQ[4] = bfloat16_to_float32(qptr4[d]) * scale;
            pQ[5] = bfloat16_to_float32(qptr5[d]) * scale;
            pQ[6] = bfloat16_to_float32(qptr6[d]) * scale;
            pQ[7] = bfloat16_to_float32(qptr7[d]) * scale;
            pQ[8] = bfloat16_to_float32(qptr8[d]) * scale;
            pQ[9] = bfloat16_to_float32(qptr9[d]) * scale;
            pQ[10] = bfloat16_to_float32(qptra[d]) * scale;
            pQ[11] = bfloat16_to_float32(qptrb[d]) * scale;
            pQ[12] = bfloat16_to_float32(qptrc[d]) * scale;
            pQ[13] = bfloat16_to_float32(qptrd[d]) * scale;
            pQ[14] = bfloat16_to_float32(qptre[d]) * scale;
            pQ[15] = bfloat16_to_float32(qptrf[d]) * scale;
            pQ += 16;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const int i0 = i + ii;
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        const unsigned short* qptr0 = query_head.row<const unsigned short>(i0);
        const unsigned short* qptr1 = query_head.row<const unsigned short>(i0 + 1);
        const unsigned short* qptr2 = query_head.row<const unsigned short>(i0 + 2);
        const unsigned short* qptr3 = query_head.row<const unsigned short>(i0 + 3);
        const unsigned short* qptr4 = query_head.row<const unsigned short>(i0 + 4);
        const unsigned short* qptr5 = query_head.row<const unsigned short>(i0 + 5);
        const unsigned short* qptr6 = query_head.row<const unsigned short>(i0 + 6);
        const unsigned short* qptr7 = query_head.row<const unsigned short>(i0 + 7);

        const __m256 _scale = _mm256_set1_ps(scale);
        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr0 + d)));
            __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr1 + d)));
            __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr2 + d)));
            __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr3 + d)));
            __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr4 + d)));
            __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr5 + d)));
            __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr6 + d)));
            __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(qptr7 + d)));
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
            pQ[0] = bfloat16_to_float32(qptr0[d]) * scale;
            pQ[1] = bfloat16_to_float32(qptr1[d]) * scale;
            pQ[2] = bfloat16_to_float32(qptr2[d]) * scale;
            pQ[3] = bfloat16_to_float32(qptr3[d]) * scale;
            pQ[4] = bfloat16_to_float32(qptr4[d]) * scale;
            pQ[5] = bfloat16_to_float32(qptr5[d]) * scale;
            pQ[6] = bfloat16_to_float32(qptr6[d]) * scale;
            pQ[7] = bfloat16_to_float32(qptr7[d]) * scale;
            pQ += 8;
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int i0 = i + ii;
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        const unsigned short* qptr0 = query_head.row<const unsigned short>(i0);
        const unsigned short* qptr1 = query_head.row<const unsigned short>(i0 + 1);
        const unsigned short* qptr2 = query_head.row<const unsigned short>(i0 + 2);
        const unsigned short* qptr3 = query_head.row<const unsigned short>(i0 + 3);

        const __m128 _scale = _mm_set1_ps(scale);
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(qptr0 + d)));
            __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(qptr1 + d)));
            __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(qptr2 + d)));
            __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(qptr3 + d)));
            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _mm_storeu_ps(pQ, _mm_mul_ps(_r0, _scale));
            _mm_storeu_ps(pQ + 4, _mm_mul_ps(_r1, _scale));
            _mm_storeu_ps(pQ + 8, _mm_mul_ps(_r2, _scale));
            _mm_storeu_ps(pQ + 12, _mm_mul_ps(_r3, _scale));
            pQ += 16;
        }
        for (; d < head_dim; d++)
        {
            pQ[0] = bfloat16_to_float32(qptr0[d]) * scale;
            pQ[1] = bfloat16_to_float32(qptr1[d]) * scale;
            pQ[2] = bfloat16_to_float32(qptr2[d]) * scale;
            pQ[3] = bfloat16_to_float32(qptr3[d]) * scale;
            pQ += 4;
        }
    }
#endif // __SSE2__
}

// packed_mask[mask_head][query_block][query_panel][key][query_lane] in fp32
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
        if (attn_mask_blob.elembits() == 32)
            sdpa_pack_query_fp32(mask_head, maskT, i0, max_ii, 1.f);
        else
            sdpa_pack_query_bf16s(mask_head, maskT, i0, max_ii, 1.f);
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
        queryT = workspace.range((block_n + value_dim) * max_ii, head_dim * max_ii);
        sdpa_pack_query_bf16s(query_head, queryT, i0, max_ii, scale);
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
                const unsigned short* key = key_head.row<const unsigned short>(n);
                __m512 _max = _mm512_set1_ps(-FLT_MAX);
                int j = 0;
                for (; j + 7 < max_jj; j += 8)
                {
                    const float* pQ = queryT;
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
                    _max = _mm512_max_ps(_max, _sum0);
                    _max = _mm512_max_ps(_max, _sum1);
                    _max = _mm512_max_ps(_max, _sum2);
                    _max = _mm512_max_ps(_max, _sum3);
                    _max = _mm512_max_ps(_max, _sum4);
                    _max = _mm512_max_ps(_max, _sum5);
                    _max = _mm512_max_ps(_max, _sum6);
                    _max = _mm512_max_ps(_max, _sum7);
                    _mm512_storeu_ps(scoreptr, _sum0);
                    _mm512_storeu_ps(scoreptr + 16, _sum1);
                    _mm512_storeu_ps(scoreptr + 32, _sum2);
                    _mm512_storeu_ps(scoreptr + 48, _sum3);
                    _mm512_storeu_ps(scoreptr + 64, _sum4);
                    _mm512_storeu_ps(scoreptr + 80, _sum5);
                    _mm512_storeu_ps(scoreptr + 96, _sum6);
                    _mm512_storeu_ps(scoreptr + 112, _sum7);
                    scoreptr += 128;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    const unsigned short* pK = key + (size_t)j * head_dim;
                    __m512 _sum = _mm512_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm512_loadu_ps(pM);
                        pM += 16;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm512_fmadd_ps(_mm512_loadu_ps(pQ), _mm512_set1_ps(bfloat16_to_float32(pK[d])), _sum);
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
                    _max = _mm512_max_ps(_max, _sum0);
                    _max = _mm512_max_ps(_max, _sum1);
                    _max = _mm512_max_ps(_max, _sum2);
                    _max = _mm512_max_ps(_max, _sum3);
                    _max = _mm512_max_ps(_max, _sum4);
                    _max = _mm512_max_ps(_max, _sum5);
                    _max = _mm512_max_ps(_max, _sum6);
                    _max = _mm512_max_ps(_max, _sum7);
                    _max = _mm512_max_ps(_max, _sum8);
                    _max = _mm512_max_ps(_max, _sum9);
                    _max = _mm512_max_ps(_max, _suma);
                    _max = _mm512_max_ps(_max, _sumb);
                    _max = _mm512_max_ps(_max, _sumc);
                    _max = _mm512_max_ps(_max, _sumd);
                    _max = _mm512_max_ps(_max, _sume);
                    _max = _mm512_max_ps(_max, _sumf);
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
                    _max = _mm512_max_ps(_max, _sum0);
                    _max = _mm512_max_ps(_max, _sum1);
                    _max = _mm512_max_ps(_max, _sum2);
                    _max = _mm512_max_ps(_max, _sum3);
                    _max = _mm512_max_ps(_max, _sum4);
                    _max = _mm512_max_ps(_max, _sum5);
                    _max = _mm512_max_ps(_max, _sum6);
                    _max = _mm512_max_ps(_max, _sum7);
                    _mm512_storeu_ps(scoreptr, _sum0);
                    _mm512_storeu_ps(scoreptr + 16, _sum1);
                    _mm512_storeu_ps(scoreptr + 32, _sum2);
                    _mm512_storeu_ps(scoreptr + 48, _sum3);
                    _mm512_storeu_ps(scoreptr + 64, _sum4);
                    _mm512_storeu_ps(scoreptr + 80, _sum5);
                    _mm512_storeu_ps(scoreptr + 96, _sum6);
                    _mm512_storeu_ps(scoreptr + 112, _sum7);
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
                    _max = _mm512_max_ps(_max, _sum0);
                    _max = _mm512_max_ps(_max, _sum1);
                    _max = _mm512_max_ps(_max, _sum2);
                    _max = _mm512_max_ps(_max, _sum3);
                    _mm512_storeu_ps(scoreptr, _sum0);
                    _mm512_storeu_ps(scoreptr + 16, _sum1);
                    _mm512_storeu_ps(scoreptr + 32, _sum2);
                    _mm512_storeu_ps(scoreptr + 48, _sum3);
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
            __m512 _sum = _mm512_setzero_ps();
            for (int j = 0; j < max_jj; j++)
            {
                __m512 _score = _mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new);
                __m512 _p = exp512_ps(_score);
                _mm512_storeu_ps(scoreptr, _p);
                scoreptr += 16;
                _sum = _mm512_add_ps(_sum, _p);
            }
            _l = _mm512_add_ps(_mm512_mul_ps(_l, _alpha), _sum);
            _m = _m_new;
            float* outptr = outT;
            if (packed_value.empty())
            {
                const unsigned short* value = value_head.row<const unsigned short>(n);
                const unsigned short* valueptr = value;
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
                const unsigned short* key = key_head.row<const unsigned short>(n);
                __m256 _max = _mm256_set1_ps(-FLT_MAX);
                int j = 0;
                for (; j + 7 < max_jj; j += 8)
                {
                    const float* pQ = queryT;
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
                    _max = _mm256_max_ps(_max, _sum0);
                    _max = _mm256_max_ps(_max, _sum1);
                    _max = _mm256_max_ps(_max, _sum2);
                    _max = _mm256_max_ps(_max, _sum3);
                    _max = _mm256_max_ps(_max, _sum4);
                    _max = _mm256_max_ps(_max, _sum5);
                    _max = _mm256_max_ps(_max, _sum6);
                    _max = _mm256_max_ps(_max, _sum7);
                    _mm256_storeu_ps(scoreptr, _sum0);
                    _mm256_storeu_ps(scoreptr + 8, _sum1);
                    _mm256_storeu_ps(scoreptr + 16, _sum2);
                    _mm256_storeu_ps(scoreptr + 24, _sum3);
                    _mm256_storeu_ps(scoreptr + 32, _sum4);
                    _mm256_storeu_ps(scoreptr + 40, _sum5);
                    _mm256_storeu_ps(scoreptr + 48, _sum6);
                    _mm256_storeu_ps(scoreptr + 56, _sum7);
                    scoreptr += 64;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    const unsigned short* pK = key + (size_t)j * head_dim;
                    __m256 _sum = _mm256_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm256_loadu_ps(pM);
                        pM += 8;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ), _mm256_set1_ps(bfloat16_to_float32(pK[d])), _sum);
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
                    for (int jj = 0; jj < 16; jj += 8)
                    {
                        const float* pK0 = pK + jj;
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
                            _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK0[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK0[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK0[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK0[3]), _sum3);
                            _sum4 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK0[4]), _sum4);
                            _sum5 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK0[5]), _sum5);
                            _sum6 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK0[6]), _sum6);
                            _sum7 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK0[7]), _sum7);
                            pQ += 8;
                            pK0 += 16;
                        }
                        _max = _mm256_max_ps(_max, _sum0);
                        _max = _mm256_max_ps(_max, _sum1);
                        _max = _mm256_max_ps(_max, _sum2);
                        _max = _mm256_max_ps(_max, _sum3);
                        _max = _mm256_max_ps(_max, _sum4);
                        _max = _mm256_max_ps(_max, _sum5);
                        _max = _mm256_max_ps(_max, _sum6);
                        _max = _mm256_max_ps(_max, _sum7);
                        _mm256_storeu_ps(scoreptr, _sum0);
                        _mm256_storeu_ps(scoreptr + 8, _sum1);
                        _mm256_storeu_ps(scoreptr + 16, _sum2);
                        _mm256_storeu_ps(scoreptr + 24, _sum3);
                        _mm256_storeu_ps(scoreptr + 32, _sum4);
                        _mm256_storeu_ps(scoreptr + 40, _sum5);
                        _mm256_storeu_ps(scoreptr + 48, _sum6);
                        _mm256_storeu_ps(scoreptr + 56, _sum7);
                        scoreptr += 64;
                    }
                    pK += (size_t)head_dim * 16;
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
                    _max = _mm256_max_ps(_max, _sum0);
                    _max = _mm256_max_ps(_max, _sum1);
                    _max = _mm256_max_ps(_max, _sum2);
                    _max = _mm256_max_ps(_max, _sum3);
                    _max = _mm256_max_ps(_max, _sum4);
                    _max = _mm256_max_ps(_max, _sum5);
                    _max = _mm256_max_ps(_max, _sum6);
                    _max = _mm256_max_ps(_max, _sum7);
                    _mm256_storeu_ps(scoreptr, _sum0);
                    _mm256_storeu_ps(scoreptr + 8, _sum1);
                    _mm256_storeu_ps(scoreptr + 16, _sum2);
                    _mm256_storeu_ps(scoreptr + 24, _sum3);
                    _mm256_storeu_ps(scoreptr + 32, _sum4);
                    _mm256_storeu_ps(scoreptr + 40, _sum5);
                    _mm256_storeu_ps(scoreptr + 48, _sum6);
                    _mm256_storeu_ps(scoreptr + 56, _sum7);
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
                    _max = _mm256_max_ps(_max, _sum0);
                    _max = _mm256_max_ps(_max, _sum1);
                    _max = _mm256_max_ps(_max, _sum2);
                    _max = _mm256_max_ps(_max, _sum3);
                    _mm256_storeu_ps(scoreptr, _sum0);
                    _mm256_storeu_ps(scoreptr + 8, _sum1);
                    _mm256_storeu_ps(scoreptr + 16, _sum2);
                    _mm256_storeu_ps(scoreptr + 24, _sum3);
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
            __m256 _sum = _mm256_setzero_ps();
            for (int j = 0; j < max_jj; j++)
            {
                __m256 _score = _mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new);
                __m256 _p = exp256_ps(_score);
                _mm256_storeu_ps(scoreptr, _p);
                scoreptr += 8;
                _sum = _mm256_add_ps(_sum, _p);
            }
            _l = _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _sum);
            _m = _m_new;
            float* outptr = outT;
            if (packed_value.empty())
            {
                const unsigned short* value = value_head.row<const unsigned short>(n);
                const unsigned short* valueptr = value;
                int d = 0;
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
                const unsigned short* key = key_head.row<const unsigned short>(n);
                __m128 _max = _mm_set1_ps(-FLT_MAX);
                int j = 0;
                for (; j + 3 < max_jj; j += 4)
                {
                    const float* pQ = queryT;
                    const unsigned short* pK0 = key + (size_t)j * head_dim;
                    const unsigned short* pK1 = pK0 + head_dim;
                    const unsigned short* pK2 = pK1 + head_dim;
                    const unsigned short* pK3 = pK2 + head_dim;
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
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK0++)), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK1++)), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK2++)), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK3++)), _sum3);
                        pQ += 4;
                    }
                    _max = _mm_max_ps(_max, _sum0);
                    _max = _mm_max_ps(_max, _sum1);
                    _max = _mm_max_ps(_max, _sum2);
                    _max = _mm_max_ps(_max, _sum3);
                    _mm_storeu_ps(scoreptr, _sum0);
                    _mm_storeu_ps(scoreptr + 4, _sum1);
                    _mm_storeu_ps(scoreptr + 8, _sum2);
                    _mm_storeu_ps(scoreptr + 12, _sum3);
                    scoreptr += 16;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    const unsigned short* pK = key + (size_t)j * head_dim;
                    __m128 _sum = _mm_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm_loadu_ps(pM);
                        pM += 4;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ), _mm_set1_ps(bfloat16_to_float32(pK[d])), _sum);
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
#if __AVX512F__
                for (; j + 15 < max_jj; j += 16)
                {
                    for (int jj = 0; jj < 16; jj += 4)
                    {
                        const float* pK0 = pK + jj;
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
                            _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK0[0]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK0[1]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK0[2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK0[3]), _sum3);
                            pQ += 4;
                            pK0 += 16;
                        }
                        _max = _mm_max_ps(_max, _sum0);
                        _max = _mm_max_ps(_max, _sum1);
                        _max = _mm_max_ps(_max, _sum2);
                        _max = _mm_max_ps(_max, _sum3);
                        _mm_storeu_ps(scoreptr, _sum0);
                        _mm_storeu_ps(scoreptr + 4, _sum1);
                        _mm_storeu_ps(scoreptr + 8, _sum2);
                        _mm_storeu_ps(scoreptr + 12, _sum3);
                        scoreptr += 16;
                    }
                    pK += (size_t)head_dim * 16;
                }
#endif // __AVX512F__
#if __AVX__
                for (; j + 7 < max_jj; j += 8)
                {
                    for (int jj = 0; jj < 8; jj += 4)
                    {
                        const float* pK0 = pK + jj;
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
                            _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK0[0]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK0[1]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK0[2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK0[3]), _sum3);
                            pQ += 4;
                            pK0 += 8;
                        }
                        _max = _mm_max_ps(_max, _sum0);
                        _max = _mm_max_ps(_max, _sum1);
                        _max = _mm_max_ps(_max, _sum2);
                        _max = _mm_max_ps(_max, _sum3);
                        _mm_storeu_ps(scoreptr, _sum0);
                        _mm_storeu_ps(scoreptr + 4, _sum1);
                        _mm_storeu_ps(scoreptr + 8, _sum2);
                        _mm_storeu_ps(scoreptr + 12, _sum3);
                        scoreptr += 16;
                    }
                    pK += (size_t)head_dim * 8;
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
                    _max = _mm_max_ps(_max, _sum0);
                    _max = _mm_max_ps(_max, _sum1);
                    _max = _mm_max_ps(_max, _sum2);
                    _max = _mm_max_ps(_max, _sum3);
                    _mm_storeu_ps(scoreptr, _sum0);
                    _mm_storeu_ps(scoreptr + 4, _sum1);
                    _mm_storeu_ps(scoreptr + 8, _sum2);
                    _mm_storeu_ps(scoreptr + 12, _sum3);
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
            __m128 _sum = _mm_setzero_ps();
            for (int j = 0; j < max_jj; j++)
            {
                __m128 _score = _mm_sub_ps(_mm_loadu_ps(scoreptr), _m_new);
                __m128 _p = exp_ps(_score);
                _mm_storeu_ps(scoreptr, _p);
                scoreptr += 4;
                _sum = _mm_add_ps(_sum, _p);
            }
            _l = _mm_add_ps(_mm_mul_ps(_l, _alpha), _sum);
            _m = _m_new;

            float* outptr = outT;
            if (packed_value.empty())
            {
                const unsigned short* value = value_head.row<const unsigned short>(n);
                const unsigned short* valueptr = value;
                int d = 0;
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
#if __AVX__
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
        const bool mask_fp32 = !mask_head.empty() && mask_head.elembits() == 32;
        const float* mask32 = mask_fp32 ? mask_head.row(i0x) : 0;
        const unsigned short* mask16 = !mask_head.empty() && !mask_fp32 ? mask_head.row<const unsigned short>(i0x) : 0;
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
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__

                float s = sum * scale;
                if (mask32)
                    s += mask32[n + j];
                else if (mask16)
                    s += bfloat16_to_float32(mask16[n + j]);
                score[j] = s;
                block_max = std::max(block_max, s);
            }

            float m_new = std::max(m, block_max);
            float alpha = l == 0.f ? 0.f : expf(m - m_new);
            float block_sum = sdpa_exp_submax_fp32(score, max_jj, m_new);
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
#endif // __AVX512F__
#if !__AVX512F__
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
#endif // !__AVX512F__

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
#endif // __AVX__
#if !__AVX__
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
#endif // !__AVX__
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
                sdpa_normalize_fp32(output_ptr, l, value_dim);
        }
    }
}
static int sdpa_prefill_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
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
    int value_pack_reuse = 4;
#if __AVX__
    value_pack_reuse = 3;
#endif // __AVX__
    if (value_dim < 32)
        value_pack_reuse += 2;
    const bool use_packed_value = key_reuse >= value_pack_reuse;
    const int block_n = sdpa_prefill_block_n(query.w, value_dim, key_seqlen, query_seqlen, use_packed_key ? 4 : 2, use_packed_value ? 4 : 2, use_packed_mask ? 4 : 0, block_m);
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
        packed_mask.create(key_seqlen * block_m, num_mblocks, num_mask_heads, 4u, opt.workspace_allocator);
        if (packed_mask.empty())
            return -100;

        sdpa_pack_mask_bf16s(attn_mask_blob, packed_mask, block_m, opt);
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
            sdpa_pack_query_bf16s(query_head, queryT, i0, max_ii, scale);
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
        sdpa_flash_attention_tile_bf16s(query, key, packed_key, value, packed_value, attn_mask_blob, packed_mask_tile, top_blob, scale, q, g, i0, max_ii, n_begin, n_end, block_n, state_stride, packed_query_tile, workspace_tile, state);
    }

    if (num_kv_chunks > 1)
    {
        sdpa_prefill_reduce(partials, top_blob, workspace, num_tasks, num_mblocks, block_m, num_kv_chunks, query_seqlen, value_dim, opt);
    }

    return 0;
}

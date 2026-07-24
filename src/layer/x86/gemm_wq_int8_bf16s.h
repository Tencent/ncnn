// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    const int elempack = A.elempack;

    if (input_scales.empty())
    {
        signed char* pp = AT_tile;
        float* pd = AT_descales_tile;
        const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
        const int block_count = (max_kk + block_size - 1) / block_size;

        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                if (elempack == 16)
                {
                    const unsigned short* p0a = p0;
                    __m512 _absmax = _mm512_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p));
                        p0a += 16;
                    }

                    __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                    __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                    __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                    _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                    __m512i _w_shift = _mm512_setzero_si512();
                    __m512i _v127 = _mm512_set1_epi8(127);
#endif
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _scale));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 32))), _scale));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 48))), _scale));
#if __AVX512VNNI__
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                        _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                        pp += 64;
                        p0 += 64;
                    }
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _mm512_storeu_si512((__m512i*)pp, _w_shift);
                        pp += 64;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _scale));
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                        p0 += 32;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                        _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                        pp += 16;
                        p0 += 16;
                    }

                    pd += 16;
                }
                if (elempack == 8)
                {
                    const unsigned short* p0a = p0;
                    __m512 _absmax = _mm512_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m512 _p;
                        _p = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a)), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 8))));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p));
                        p0a += 8;
                    }
                    __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                    __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                    __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                    _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                    __m512i _w_shift = _mm512_setzero_si512();
                    __m512i _v127 = _mm512_set1_epi8(127);
#endif
                    int kk = 0;
#if __AVX512VNNI__
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m512 _p0, _p1, _p2, _p3;

                        _p0 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 0 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 0 * 8))));
                        _p1 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 1 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 1 * 8))));
                        _p2 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 2 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 2 * 8))));
                        _p3 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 3 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 3 * 8))));
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_p2, _scale));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_p3, _scale));
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                        p0 += 8 * 4;
                    }
#endif
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _mm512_storeu_si512((__m512i*)pp, _w_shift);
                        pp += 64;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m512 _p0;
                        __m512 _p1;

                        _p0 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8))));
                        _p1 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 8))));

                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                        p0 += 8 * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m512 _p = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8))));
                        _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                        pp += 16;
                        p0 += 8;
                    }
                    pd += 16;
                }
                if (elempack == 4)
                {
                    const unsigned short* p0a = p0;
                    __m512 _absmax = _mm512_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m512 _p;
                        _p = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 8))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 12))));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p));
                        p0a += 4;
                    }
                    __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                    __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                    __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                    _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                    __m512i _w_shift = _mm512_setzero_si512();
                    __m512i _v127 = _mm512_set1_epi8(127);
#endif
                    int kk = 0;
#if __AVX512VNNI__
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m512 _p0, _p1, _p2, _p3;

                        _p0 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 0 * 4))));
                        _p1 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 1 * 4))));
                        _p2 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 2 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 2 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 2 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 2 * 4))));
                        _p3 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 3 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 3 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 3 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 3 * 4))));
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_p2, _scale));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_p3, _scale));
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                        p0 += 4 * 4;
                    }
#endif
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _mm512_storeu_si512((__m512i*)pp, _w_shift);
                        pp += 64;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m512 _p0;
                        __m512 _p1;

                        _p0 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12))));
                        _p1 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 4))));

                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                        p0 += 4 * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m512 _p = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12))));
                        _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                        pp += 16;
                        p0 += 4;
                    }
                    pd += 16;
                }
                if (elempack == 1)
                {
                    const unsigned short* p0a = p0;

                    __m512 _absmax0 = _mm512_setzero_ps();
                    __m512 _absmax1 = _mm512_setzero_ps();
                    __m512 _absmax2 = _mm512_setzero_ps();
                    __m512 _absmax3 = _mm512_setzero_ps();
                    __m512 _absmax4 = _mm512_setzero_ps();
                    __m512 _absmax5 = _mm512_setzero_ps();
                    __m512 _absmax6 = _mm512_setzero_ps();
                    __m512 _absmax7 = _mm512_setzero_ps();
                    __m512 _absmax8 = _mm512_setzero_ps();
                    __m512 _absmax9 = _mm512_setzero_ps();
                    __m512 _absmaxa = _mm512_setzero_ps();
                    __m512 _absmaxb = _mm512_setzero_ps();
                    __m512 _absmaxc = _mm512_setzero_ps();
                    __m512 _absmaxd = _mm512_setzero_ps();
                    __m512 _absmaxe = _mm512_setzero_ps();
                    __m512 _absmaxf = _mm512_setzero_ps();
                    int kk = 0;
                    for (; kk + 15 < max_kk0; kk += 16)
                    {
                        __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a)));
                        _absmax0 = _mm512_max_ps(_absmax0, abs512_ps(_p0));
                        __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep)));
                        _absmax1 = _mm512_max_ps(_absmax1, abs512_ps(_p1));
                        __m512 _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 2)));
                        _absmax2 = _mm512_max_ps(_absmax2, abs512_ps(_p2));
                        __m512 _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 3)));
                        _absmax3 = _mm512_max_ps(_absmax3, abs512_ps(_p3));
                        __m512 _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 4)));
                        _absmax4 = _mm512_max_ps(_absmax4, abs512_ps(_p4));
                        __m512 _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 5)));
                        _absmax5 = _mm512_max_ps(_absmax5, abs512_ps(_p5));
                        __m512 _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 6)));
                        _absmax6 = _mm512_max_ps(_absmax6, abs512_ps(_p6));
                        __m512 _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 7)));
                        _absmax7 = _mm512_max_ps(_absmax7, abs512_ps(_p7));
                        __m512 _p8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 8)));
                        _absmax8 = _mm512_max_ps(_absmax8, abs512_ps(_p8));
                        __m512 _p9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 9)));
                        _absmax9 = _mm512_max_ps(_absmax9, abs512_ps(_p9));
                        __m512 _pa = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 10)));
                        _absmaxa = _mm512_max_ps(_absmaxa, abs512_ps(_pa));
                        __m512 _pb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 11)));
                        _absmaxb = _mm512_max_ps(_absmaxb, abs512_ps(_pb));
                        __m512 _pc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 12)));
                        _absmaxc = _mm512_max_ps(_absmaxc, abs512_ps(_pc));
                        __m512 _pd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 13)));
                        _absmaxd = _mm512_max_ps(_absmaxd, abs512_ps(_pd));
                        __m512 _pe = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 14)));
                        _absmaxe = _mm512_max_ps(_absmaxe, abs512_ps(_pe));
                        __m512 _pf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 15)));
                        _absmaxf = _mm512_max_ps(_absmaxf, abs512_ps(_pf));
                        p0a += 16;
                    }

                    float absmax0 = _mm512_reduce_max_ps(_absmax0);
                    float absmax1 = _mm512_reduce_max_ps(_absmax1);
                    float absmax2 = _mm512_reduce_max_ps(_absmax2);
                    float absmax3 = _mm512_reduce_max_ps(_absmax3);
                    float absmax4 = _mm512_reduce_max_ps(_absmax4);
                    float absmax5 = _mm512_reduce_max_ps(_absmax5);
                    float absmax6 = _mm512_reduce_max_ps(_absmax6);
                    float absmax7 = _mm512_reduce_max_ps(_absmax7);
                    float absmax8 = _mm512_reduce_max_ps(_absmax8);
                    float absmax9 = _mm512_reduce_max_ps(_absmax9);
                    float absmaxa = _mm512_reduce_max_ps(_absmaxa);
                    float absmaxb = _mm512_reduce_max_ps(_absmaxb);
                    float absmaxc = _mm512_reduce_max_ps(_absmaxc);
                    float absmaxd = _mm512_reduce_max_ps(_absmaxd);
                    float absmaxe = _mm512_reduce_max_ps(_absmaxe);
                    float absmaxf = _mm512_reduce_max_ps(_absmaxf);
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                        absmax0 = std::max(absmax0, _mm_reduce_max_ps(abs_ps(_p0)));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep)));
                        absmax1 = std::max(absmax1, _mm_reduce_max_ps(abs_ps(_p1)));
                        __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 2)));
                        absmax2 = std::max(absmax2, _mm_reduce_max_ps(abs_ps(_p2)));
                        __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 3)));
                        absmax3 = std::max(absmax3, _mm_reduce_max_ps(abs_ps(_p3)));
                        __m128 _p4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 4)));
                        absmax4 = std::max(absmax4, _mm_reduce_max_ps(abs_ps(_p4)));
                        __m128 _p5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 5)));
                        absmax5 = std::max(absmax5, _mm_reduce_max_ps(abs_ps(_p5)));
                        __m128 _p6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 6)));
                        absmax6 = std::max(absmax6, _mm_reduce_max_ps(abs_ps(_p6)));
                        __m128 _p7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 7)));
                        absmax7 = std::max(absmax7, _mm_reduce_max_ps(abs_ps(_p7)));
                        __m128 _p8 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 8)));
                        absmax8 = std::max(absmax8, _mm_reduce_max_ps(abs_ps(_p8)));
                        __m128 _p9 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 9)));
                        absmax9 = std::max(absmax9, _mm_reduce_max_ps(abs_ps(_p9)));
                        __m128 _pa = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 10)));
                        absmaxa = std::max(absmaxa, _mm_reduce_max_ps(abs_ps(_pa)));
                        __m128 _pb = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 11)));
                        absmaxb = std::max(absmaxb, _mm_reduce_max_ps(abs_ps(_pb)));
                        __m128 _pc = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 12)));
                        absmaxc = std::max(absmaxc, _mm_reduce_max_ps(abs_ps(_pc)));
                        __m128 _pd = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 13)));
                        absmaxd = std::max(absmaxd, _mm_reduce_max_ps(abs_ps(_pd)));
                        __m128 _pe = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 14)));
                        absmaxe = std::max(absmaxe, _mm_reduce_max_ps(abs_ps(_pe)));
                        __m128 _pf = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 15)));
                        absmaxf = std::max(absmaxf, _mm_reduce_max_ps(abs_ps(_pf)));
                        p0a += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])));
                        absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[A_hstep])));
                        absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(p0a[A_hstep * 2])));
                        absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(p0a[A_hstep * 3])));
                        absmax4 = std::max(absmax4, fabsf(bfloat16_to_float32(p0a[A_hstep * 4])));
                        absmax5 = std::max(absmax5, fabsf(bfloat16_to_float32(p0a[A_hstep * 5])));
                        absmax6 = std::max(absmax6, fabsf(bfloat16_to_float32(p0a[A_hstep * 6])));
                        absmax7 = std::max(absmax7, fabsf(bfloat16_to_float32(p0a[A_hstep * 7])));
                        absmax8 = std::max(absmax8, fabsf(bfloat16_to_float32(p0a[A_hstep * 8])));
                        absmax9 = std::max(absmax9, fabsf(bfloat16_to_float32(p0a[A_hstep * 9])));
                        absmaxa = std::max(absmaxa, fabsf(bfloat16_to_float32(p0a[A_hstep * 10])));
                        absmaxb = std::max(absmaxb, fabsf(bfloat16_to_float32(p0a[A_hstep * 11])));
                        absmaxc = std::max(absmaxc, fabsf(bfloat16_to_float32(p0a[A_hstep * 12])));
                        absmaxd = std::max(absmaxd, fabsf(bfloat16_to_float32(p0a[A_hstep * 13])));
                        absmaxe = std::max(absmaxe, fabsf(bfloat16_to_float32(p0a[A_hstep * 14])));
                        absmaxf = std::max(absmaxf, fabsf(bfloat16_to_float32(p0a[A_hstep * 15])));
                        p0a++;
                    }
                    __m512 _absmax = _mm512_setr_ps(absmax0, absmax1, absmax2, absmax3, absmax4, absmax5, absmax6, absmax7, absmax8, absmax9, absmaxa, absmaxb, absmaxc, absmaxd, absmaxe, absmaxf);

                    __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                    __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                    __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                    _mm512_storeu_ps(pd, _descale);

#if __AVX512VNNI__
                    __m512i _w_shift = _mm512_setzero_si512();
                    __m512i _v127 = _mm512_set1_epi8(127);
#endif
                    kk = 0;
                    for (; kk + 15 < max_kk0; kk += 16)
                    {
                        {
                            __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                            __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep)));
                            __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2)));
                            __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3)));
                            __m256 _p4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4)));
                            __m256 _p5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 5)));
                            __m256 _p6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 6)));
                            __m256 _p7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 7)));
                            __m256 _p8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8)));
                            __m256 _p9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 9)));
                            __m256 _pa = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 10)));
                            __m256 _pb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 11)));
                            __m256 _pc = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 12)));
                            __m256 _pd = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 13)));
                            __m256 _pe = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 14)));
                            __m256 _pf = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 15)));
                            transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                            transpose8x8_ps(_p8, _p9, _pa, _pb, _pc, _pd, _pe, _pf);

                            __m512 _t0 = combine8x2_ps(_p0, _p8);
                            __m512 _t1 = combine8x2_ps(_p1, _p9);
                            __m512 _t2 = combine8x2_ps(_p2, _pa);
                            __m512 _t3 = combine8x2_ps(_p3, _pb);
                            __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                            __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                            __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                            __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                            transpose16x4_epi8(_q0, _q1, _q2, _q3);
                            __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                            _mm512_storeu_si512((__m512i*)pp, _q);
                            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                            _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                            pp += 64;

                            _t0 = combine8x2_ps(_p4, _pc);
                            _t1 = combine8x2_ps(_p5, _pd);
                            _t2 = combine8x2_ps(_p6, _pe);
                            _t3 = combine8x2_ps(_p7, _pf);
                            _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                            _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                            _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                            _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                            transpose16x4_epi8(_q0, _q1, _q2, _q3);
                            _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                            _mm512_storeu_si512((__m512i*)pp, _q);
                            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                            _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                            pp += 64;
                        }
                        {
                            __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
                            __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep + 8)));
                            __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2 + 8)));
                            __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3 + 8)));
                            __m256 _p4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4 + 8)));
                            __m256 _p5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 5 + 8)));
                            __m256 _p6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 6 + 8)));
                            __m256 _p7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 7 + 8)));
                            __m256 _p8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 8)));
                            __m256 _p9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 9 + 8)));
                            __m256 _pa = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 10 + 8)));
                            __m256 _pb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 11 + 8)));
                            __m256 _pc = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 12 + 8)));
                            __m256 _pd = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 13 + 8)));
                            __m256 _pe = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 14 + 8)));
                            __m256 _pf = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 15 + 8)));
                            transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                            transpose8x8_ps(_p8, _p9, _pa, _pb, _pc, _pd, _pe, _pf);

                            __m512 _t0 = combine8x2_ps(_p0, _p8);
                            __m512 _t1 = combine8x2_ps(_p1, _p9);
                            __m512 _t2 = combine8x2_ps(_p2, _pa);
                            __m512 _t3 = combine8x2_ps(_p3, _pb);
                            __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                            __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                            __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                            __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                            transpose16x4_epi8(_q0, _q1, _q2, _q3);
                            __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                            _mm512_storeu_si512((__m512i*)pp, _q);
                            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                            _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                            pp += 64;

                            _t0 = combine8x2_ps(_p4, _pc);
                            _t1 = combine8x2_ps(_p5, _pd);
                            _t2 = combine8x2_ps(_p6, _pe);
                            _t3 = combine8x2_ps(_p7, _pf);
                            _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                            _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                            _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                            _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                            transpose16x4_epi8(_q0, _q1, _q2, _q3);
                            _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                            _mm512_storeu_si512((__m512i*)pp, _q);
                            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                            _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                            pp += 64;
                        }
                        p0 += 16;
                    }
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                        __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                        __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                        __m128 _p4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4)));
                        __m128 _p5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 5)));
                        __m128 _p6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 6)));
                        __m128 _p7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 7)));
                        __m128 _p8 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8)));
                        __m128 _p9 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 9)));
                        __m128 _pa = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 10)));
                        __m128 _pb = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 11)));
                        __m128 _pc = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12)));
                        __m128 _pd = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 13)));
                        __m128 _pe = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 14)));
                        __m128 _pf = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 15)));
                        __m512 _t0 = combine4x4_ps(_p0, _p4, _p8, _pc);
                        __m512 _t1 = combine4x4_ps(_p1, _p5, _p9, _pd);
                        __m512 _t2 = combine4x4_ps(_p2, _p6, _pa, _pe);
                        __m512 _t3 = combine4x4_ps(_p3, _p7, _pb, _pf);
                        __m512 _t4 = _mm512_unpacklo_ps(_t0, _t1);
                        __m512 _t5 = _mm512_unpackhi_ps(_t0, _t1);
                        __m512 _t6 = _mm512_unpacklo_ps(_t2, _t3);
                        __m512 _t7 = _mm512_unpackhi_ps(_t2, _t3);
                        _t0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_t4), _mm512_castps_pd(_t6)));
                        _t1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_t4), _mm512_castps_pd(_t6)));
                        _t2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_t5), _mm512_castps_pd(_t7)));
                        _t3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_t5), _mm512_castps_pd(_t7)));
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                        _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                        pp += 64;
                        p0 += 4;
                    }
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _mm512_storeu_si512((__m512i*)pp, _w_shift);
                        pp += 64;
                    }
#endif // __AVX512VNNI__
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m512 _p0 = _mm512_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]), bfloat16_to_float32(p0[A_hstep * 8]), bfloat16_to_float32(p0[A_hstep * 9]), bfloat16_to_float32(p0[A_hstep * 10]), bfloat16_to_float32(p0[A_hstep * 11]), bfloat16_to_float32(p0[A_hstep * 12]), bfloat16_to_float32(p0[A_hstep * 13]), bfloat16_to_float32(p0[A_hstep * 14]), bfloat16_to_float32(p0[A_hstep * 15]));
                        __m512 _p1 = _mm512_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), bfloat16_to_float32(p0[A_hstep * 2 + 1]), bfloat16_to_float32(p0[A_hstep * 3 + 1]), bfloat16_to_float32(p0[A_hstep * 4 + 1]), bfloat16_to_float32(p0[A_hstep * 5 + 1]), bfloat16_to_float32(p0[A_hstep * 6 + 1]), bfloat16_to_float32(p0[A_hstep * 7 + 1]), bfloat16_to_float32(p0[A_hstep * 8 + 1]), bfloat16_to_float32(p0[A_hstep * 9 + 1]), bfloat16_to_float32(p0[A_hstep * 10 + 1]), bfloat16_to_float32(p0[A_hstep * 11 + 1]), bfloat16_to_float32(p0[A_hstep * 12 + 1]), bfloat16_to_float32(p0[A_hstep * 13 + 1]), bfloat16_to_float32(p0[A_hstep * 14 + 1]), bfloat16_to_float32(p0[A_hstep * 15 + 1]));
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m512 _p = _mm512_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]), bfloat16_to_float32(p0[A_hstep * 8]), bfloat16_to_float32(p0[A_hstep * 9]), bfloat16_to_float32(p0[A_hstep * 10]), bfloat16_to_float32(p0[A_hstep * 11]), bfloat16_to_float32(p0[A_hstep * 12]), bfloat16_to_float32(p0[A_hstep * 13]), bfloat16_to_float32(p0[A_hstep * 14]), bfloat16_to_float32(p0[A_hstep * 15]));
                        _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                        pp += 16;
                        p0++;
                    }

                    pd += 16;
                }
            }
        }
#endif // __AVX512F__
#if !__AVX2__
        signed char* pp1 = pp + AT_tile.w * 4;
        float* pd1 = pd + AT_descales_tile.w * 4;
#endif
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                if (elempack == 8)
                {
                    const unsigned short* p0a = p0;
                    __m256 _absmax = _mm256_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p));
                        p0a += 8;
                    }

                    __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                    __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                    __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                    __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                    _mm256_storeu_ps(pd, _descale);
#else
                    _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                    _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m256i _w_shift = _mm256_setzero_si256();
#endif
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m256 _p0 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _scale);
                        __m256 _p1 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _scale);
                        __m256 _p2 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 16))), _scale);
                        __m256 _p3 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 24))), _scale);
                        __m128i _q0 = float2int8_avx(_p0, _p2);
                        __m128i _q1 = float2int8_avx(_p1, _p3);
                        __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                        __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                        _q0 = _mm_unpacklo_epi16(_q01, _q23);
                        _q1 = _mm_unpackhi_epi16(_q01, _q23);
                        __m256i _q = combine4x2_epi32(_q0, _q1);
                        _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                        _mm_storeu_si128((__m128i*)pp, _q01);
                        _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                        _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif
#if __AVX2__
                        pp += 32;
#else
                        pp += 16;
                        pp1 += 16;
#endif
                        p0 += 32;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm256_storeu_si256((__m256i*)pp, _w_shift);
                        pp += 32;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m256 _p0 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _scale);
                        __m256 _p1 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _scale);
                        __m128i _q = float2int8_avx(_p0, _p1);
                        __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                        _q = _mm_shuffle_epi8(_q, _si);
#if __AVX2__
                        _mm_storeu_si128((__m128i*)pp, _q);
                        pp += 16;
#else
                        _mm_storel_pd((double*)pp, _mm_castsi128_pd(_q));
                        _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q));
                        pp += 8;
                        pp1 += 8;
#endif
                        p0 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m256 _p = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _scale);
#if __AVX2__
                        _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(float2int8_avx(_p)));
                        pp += 8;
#else
                        const uint64_t q = (uint64_t)float2int8_avx(_p);
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128((int)q)));
                        _mm_store_ss((float*)pp1, _mm_castsi128_ps(_mm_cvtsi32_si128((int)(q >> 32))));
                        pp += 4;
                        pp1 += 4;
#endif
                        p0 += 8;
                    }

#if __AVX2__
                    pd += 8;
#else
                    pd += 4;
                    pd1 += 4;
#endif
                }
                if (elempack == 4)
                {
                    const unsigned short* p0a = p0;
                    __m256 _absmax = _mm256_setzero_ps();
                    int kk = 0;
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 4)));
                        __m256 _t0 = _mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 2, 0, 0));
                        __m256 _t1 = _mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 3, 0, 1));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_t0));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_t1));
                        p0a += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m256 _p = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 4))));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p));
                        p0a += 4;
                    }

                    __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                    __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                    __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                    __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                    _mm256_storeu_ps(pd, _descale);
#else
                    _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                    _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m256i _w_shift = _mm256_setzero_si256();
#endif
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
                        __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4)));
                        __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4 + 8)));
                        __m256 _t0 = _mm256_mul_ps(_mm256_permute2f128_ps(_p0, _p2, _MM_SHUFFLE(0, 2, 0, 0)), _scale);
                        __m256 _t1 = _mm256_mul_ps(_mm256_permute2f128_ps(_p0, _p2, _MM_SHUFFLE(0, 3, 0, 1)), _scale);
                        __m256 _t2 = _mm256_mul_ps(_mm256_permute2f128_ps(_p1, _p3, _MM_SHUFFLE(0, 2, 0, 0)), _scale);
                        __m256 _t3 = _mm256_mul_ps(_mm256_permute2f128_ps(_p1, _p3, _MM_SHUFFLE(0, 3, 0, 1)), _scale);
                        __m128i _q0 = float2int8_avx(_t0, _t2);
                        __m128i _q1 = float2int8_avx(_t1, _t3);
                        __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                        __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                        _q0 = _mm_unpacklo_epi16(_q01, _q23);
                        _q1 = _mm_unpackhi_epi16(_q01, _q23);
                        __m256i _q = combine4x2_epi32(_q0, _q1);
                        _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                        _mm_storeu_si128((__m128i*)pp, _q01);
                        _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                        _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif
#if __AVX2__
                        pp += 32;
#else
                        pp += 16;
                        pp1 += 16;
#endif
                        p0 += 16;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm256_storeu_si256((__m256i*)pp, _w_shift);
                        pp += 32;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4)));
                        __m256 _t0 = _mm256_mul_ps(_mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 2, 0, 0)), _scale);
                        __m256 _t1 = _mm256_mul_ps(_mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 3, 0, 1)), _scale);
                        __m128i _q = float2int8_avx(_t0, _t1);
                        __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                        _q = _mm_shuffle_epi8(_q, _si);
#if __AVX2__
                        _mm_storeu_si128((__m128i*)pp, _q);
                        pp += 16;
#else
                        _mm_storel_pd((double*)pp, _mm_castsi128_pd(_q));
                        _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q));
                        pp += 8;
                        pp1 += 8;
#endif
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m256 _p = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4))));
                        _p = _mm256_mul_ps(_p, _scale);
#if __AVX2__
                        _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(float2int8_avx(_p)));
                        pp += 8;
#else
                        const uint64_t q = (uint64_t)float2int8_avx(_p);
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128((int)q)));
                        _mm_store_ss((float*)pp1, _mm_castsi128_ps(_mm_cvtsi32_si128((int)(q >> 32))));
                        pp += 4;
                        pp1 += 4;
#endif
                        p0 += 4;
                    }

#if __AVX2__
                    pd += 8;
#else
                    pd += 4;
                    pd1 += 4;
#endif
                }
                if (elempack == 1)
                {
                    const unsigned short* p0a = p0;

                    __m256 _absmax0 = _mm256_setzero_ps();
                    __m256 _absmax1 = _mm256_setzero_ps();
                    __m256 _absmax2 = _mm256_setzero_ps();
                    __m256 _absmax3 = _mm256_setzero_ps();
                    __m256 _absmax4 = _mm256_setzero_ps();
                    __m256 _absmax5 = _mm256_setzero_ps();
                    __m256 _absmax6 = _mm256_setzero_ps();
                    __m256 _absmax7 = _mm256_setzero_ps();
                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a)));
                        _absmax0 = _mm256_max_ps(_absmax0, abs256_ps(_p0));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep)));
                        _absmax1 = _mm256_max_ps(_absmax1, abs256_ps(_p1));
                        __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 2)));
                        _absmax2 = _mm256_max_ps(_absmax2, abs256_ps(_p2));
                        __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 3)));
                        _absmax3 = _mm256_max_ps(_absmax3, abs256_ps(_p3));
                        __m256 _p4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 4)));
                        _absmax4 = _mm256_max_ps(_absmax4, abs256_ps(_p4));
                        __m256 _p5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 5)));
                        _absmax5 = _mm256_max_ps(_absmax5, abs256_ps(_p5));
                        __m256 _p6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 6)));
                        _absmax6 = _mm256_max_ps(_absmax6, abs256_ps(_p6));
                        __m256 _p7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 7)));
                        _absmax7 = _mm256_max_ps(_absmax7, abs256_ps(_p7));
                        p0a += 8;
                    }

                    float absmax0 = _mm256_reduce_max_ps(_absmax0);
                    float absmax1 = _mm256_reduce_max_ps(_absmax1);
                    float absmax2 = _mm256_reduce_max_ps(_absmax2);
                    float absmax3 = _mm256_reduce_max_ps(_absmax3);
                    float absmax4 = _mm256_reduce_max_ps(_absmax4);
                    float absmax5 = _mm256_reduce_max_ps(_absmax5);
                    float absmax6 = _mm256_reduce_max_ps(_absmax6);
                    float absmax7 = _mm256_reduce_max_ps(_absmax7);
                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])));
                        absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[A_hstep])));
                        absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(p0a[A_hstep * 2])));
                        absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(p0a[A_hstep * 3])));
                        absmax4 = std::max(absmax4, fabsf(bfloat16_to_float32(p0a[A_hstep * 4])));
                        absmax5 = std::max(absmax5, fabsf(bfloat16_to_float32(p0a[A_hstep * 5])));
                        absmax6 = std::max(absmax6, fabsf(bfloat16_to_float32(p0a[A_hstep * 6])));
                        absmax7 = std::max(absmax7, fabsf(bfloat16_to_float32(p0a[A_hstep * 7])));
                        p0a++;
                    }

                    __m256 _absmax = _mm256_setr_ps(absmax0, absmax1, absmax2, absmax3, absmax4, absmax5, absmax6, absmax7);
                    __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                    __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                    __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                    __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                    _mm256_storeu_ps(pd, _descale);
#else
                    _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                    _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m256i _w_shift = _mm256_setzero_si256();
#endif
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                        __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                        __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                        __m128 _p4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4)));
                        __m128 _p5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 5)));
                        __m128 _p6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 6)));
                        __m128 _p7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 7)));

                        __m256 _t0 = combine4x2_ps(_p0, _p4);
                        __m256 _t1 = combine4x2_ps(_p1, _p5);
                        __m256 _t2 = combine4x2_ps(_p2, _p6);
                        __m256 _t3 = combine4x2_ps(_p3, _p7);
                        __m256 _t4 = _mm256_unpacklo_ps(_t0, _t1);
                        __m256 _t5 = _mm256_unpackhi_ps(_t0, _t1);
                        __m256 _t6 = _mm256_unpacklo_ps(_t2, _t3);
                        __m256 _t7 = _mm256_unpackhi_ps(_t2, _t3);
                        _t0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_t4), _mm256_castps_pd(_t6)));
                        _t1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_t4), _mm256_castps_pd(_t6)));
                        _t2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_t5), _mm256_castps_pd(_t7)));
                        _t3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_t5), _mm256_castps_pd(_t7)));
                        _t0 = _mm256_mul_ps(_t0, _scale);
                        _t1 = _mm256_mul_ps(_t1, _scale);
                        _t2 = _mm256_mul_ps(_t2, _scale);
                        _t3 = _mm256_mul_ps(_t3, _scale);

                        __m128i _q0 = float2int8_avx(_t0, _t2);
                        __m128i _q1 = float2int8_avx(_t1, _t3);
                        __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                        __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                        _q0 = _mm_unpacklo_epi16(_q01, _q23);
                        _q1 = _mm_unpackhi_epi16(_q01, _q23);
                        __m256i _q = combine4x2_epi32(_q0, _q1);
                        _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                        _mm_storeu_si128((__m128i*)pp, _q01);
                        _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                        _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX2__
                        pp += 32;
#else
                        pp += 16;
                        pp1 += 16;
#endif
                        p0 += 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm256_storeu_si256((__m256i*)pp, _w_shift);
                        pp += 32;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
#if __AVX2__
                        __m256 _p0 = _mm256_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
                        __m256 _p1 = _mm256_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), bfloat16_to_float32(p0[A_hstep * 2 + 1]), bfloat16_to_float32(p0[A_hstep * 3 + 1]), bfloat16_to_float32(p0[A_hstep * 4 + 1]), bfloat16_to_float32(p0[A_hstep * 5 + 1]), bfloat16_to_float32(p0[A_hstep * 6 + 1]), bfloat16_to_float32(p0[A_hstep * 7 + 1]));
#else
                        __m128 _p00 = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                        __m128 _p01 = _mm_setr_ps(bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
                        __m128 _p10 = _mm_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), bfloat16_to_float32(p0[A_hstep * 2 + 1]), bfloat16_to_float32(p0[A_hstep * 3 + 1]));
                        __m128 _p11 = _mm_setr_ps(bfloat16_to_float32(p0[A_hstep * 4 + 1]), bfloat16_to_float32(p0[A_hstep * 5 + 1]), bfloat16_to_float32(p0[A_hstep * 6 + 1]), bfloat16_to_float32(p0[A_hstep * 7 + 1]));
                        __m256 _p0 = combine4x2_ps(_p00, _p01);
                        __m256 _p1 = combine4x2_ps(_p10, _p11);
#endif // __AVX2__
                        _p0 = _mm256_mul_ps(_p0, _scale);
                        _p1 = _mm256_mul_ps(_p1, _scale);
                        __m128i _q = float2int8_avx(_p0, _p1);
                        __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                        _q = _mm_shuffle_epi8(_q, _si);
#if __AVX2__
                        _mm_storeu_si128((__m128i*)pp, _q);
                        pp += 16;
#else
                        _mm_storel_pd((double*)pp, _mm_castsi128_pd(_q));
                        _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q));
                        pp += 8;
                        pp1 += 8;
#endif
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
#if __AVX2__
                        __m256 _p = _mm256_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
#else
                        __m128 _p0 = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                        __m128 _p1 = _mm_setr_ps(bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
                        __m256 _p = combine4x2_ps(_p0, _p1);
#endif // __AVX2__
#if __AVX2__
                        _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(float2int8_avx(_mm256_mul_ps(_p, _scale))));
                        pp += 8;
#else
                        const uint64_t q = (uint64_t)float2int8_avx(_mm256_mul_ps(_p, _scale));
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128((int)q)));
                        _mm_store_ss((float*)pp1, _mm_castsi128_ps(_mm_cvtsi32_si128((int)(q >> 32))));
                        pp += 4;
                        pp1 += 4;
#endif
                        p0++;
                    }

#if __AVX2__
                    pd += 8;
#else
                    pd += 4;
                    pd1 += 4;
#endif
                }
            }
#if !__AVX2__
            pp = pp1;
            pp1 = pp + AT_tile.w * 4;
            pd = pd1;
            pd1 = pd + AT_descales_tile.w * 4;
#endif
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                if (elempack == 4)
                {
                    const unsigned short* p0a = p0;
                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a));
                        _absmax = _mm_max_ps(_absmax, abs_ps(_p));
                        p0a += 4;
                    }

                    __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                    __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                    __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                    __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                    _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
                    __m128i _v127 = _mm_set1_epi8(127);
#endif
                    int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4)));
                        __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 8)));
                        __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 12)));
                        _p0 = _mm_mul_ps(_p0, _scale);
                        _p1 = _mm_mul_ps(_p1, _scale);
                        _p2 = _mm_mul_ps(_p2, _scale);
                        _p3 = _mm_mul_ps(_p3, _scale);
                        __m128i _q = float2int8_sse(_p0, _p1, _p2, _p3);
                        __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                        _q = _mm_shuffle_epi8(_q, _si);
#if !__AVXVNNIINT8__
                        _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _q);
#endif
                        _mm_storeu_si128((__m128i*)pp, _q);
                        pp += 16;
                        p0 += 16;
                    }
#if !__AVXVNNIINT8__
                    if (max_kk0 >= 4)
                    {
                        _mm_storeu_si128((__m128i*)pp, _w_shift);
                        pp += 16;
                    }
#endif
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4)));
                        _p0 = _mm_mul_ps(_p0, _scale);
                        _p1 = _mm_mul_ps(_p1, _scale);
                        __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                        __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                        *(int64_t*)pp = float2int8_sse(_t0, _t1);
                        pp += 8;
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                        _p = _mm_mul_ps(_p, _scale);
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(float2int8_sse(_p))));
                        pp += 4;
                        p0 += 4;
                    }

                    pd += 4;
                }
                if (elempack == 1)
                {
                    const unsigned short* p0a = p0;

                    __m128 _absmax0 = _mm_setzero_ps();
                    __m128 _absmax1 = _mm_setzero_ps();
                    __m128 _absmax2 = _mm_setzero_ps();
                    __m128 _absmax3 = _mm_setzero_ps();
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                        _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p0));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep)));
                        _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
                        __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 2)));
                        _absmax2 = _mm_max_ps(_absmax2, abs_ps(_p2));
                        __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 3)));
                        _absmax3 = _mm_max_ps(_absmax3, abs_ps(_p3));
                        p0a += 4;
                    }

                    float absmax0 = _mm_reduce_max_ps(_absmax0);
                    float absmax1 = _mm_reduce_max_ps(_absmax1);
                    float absmax2 = _mm_reduce_max_ps(_absmax2);
                    float absmax3 = _mm_reduce_max_ps(_absmax3);
                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])));
                        absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[A_hstep])));
                        absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(p0a[A_hstep * 2])));
                        absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(p0a[A_hstep * 3])));
                        p0a++;
                    }

                    __m128 _absmax = _mm_setr_ps(absmax0, absmax1, absmax2, absmax3);
                    __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                    __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                    __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                    __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                    _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
#endif
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                        __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                        __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                        __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(0, 0, 0, 0)))));
                        __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(1, 1, 1, 1)))));
                        __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p2, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(2, 2, 2, 2)))));
                        __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p3, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(3, 3, 3, 3)))));
#if __AVX512VNNI__ || __AVXVNNI__
                        __m128i _q = _mm_unpacklo_epi64(_mm_unpacklo_epi32(_q0, _q1), _mm_unpacklo_epi32(_q2, _q3));
                        _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                        __m128i _q01 = _mm_unpacklo_epi16(_q0, _q1);
                        __m128i _q23 = _mm_unpacklo_epi16(_q2, _q3);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi32(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
                        pp += 16;
                        p0 += 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_storeu_si128((__m128i*)pp, _w_shift);
                        pp += 16;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m128 _p0 = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                        __m128 _p1 = _mm_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), bfloat16_to_float32(p0[A_hstep * 2 + 1]), bfloat16_to_float32(p0[A_hstep * 3 + 1]));
                        __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                        __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                        _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        pp += 8;
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p, _scale)))));
                        pp += 4;
                        p0++;
                    }

                    pd += 4;
                }
            }
        }
#endif // __SSE2__
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const unsigned short* p0a = p0;

                float absmax0 = 0.f;
                float absmax1 = 0.f;
                int kk = 0;
#if __SSE2__
                __m128 _absmax0 = _mm_setzero_ps();
                __m128 _absmax1 = _mm_setzero_ps();
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                    _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p0));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep)));
                    _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
                    p0a += 4;
                }

                absmax0 = _mm_reduce_max_ps(_absmax0);
                absmax1 = _mm_reduce_max_ps(_absmax1);
#endif // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])));
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[A_hstep])));
                    p0a++;
                }

                float scale0 = 0.f;
                float scale1 = 0.f;
                if (absmax0 != 0.f)
                    scale0 = 127.f / absmax0;
                if (absmax1 != 0.f)
                    scale1 = 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                kk = 0;
#if __SSE2__
                __m128 _scale = _mm_setr_ps(scale0, scale1, 0.f, 0.f);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _w_shift = _mm_setzero_si128();
#endif
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(0, 0, 0, 0)))));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(1, 1, 1, 1)))));
#if __AVX512VNNI__ || __AVXVNNI__
                    __m128i _q = _mm_unpacklo_epi32(_q0, _q1);
                    _mm_storel_epi64((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                    _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi16(_q0, _q1));
#endif // __AVX512VNNI__ || __AVXVNNI__
                    pp += 8;
                    p0 += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_storel_epi64((__m128i*)pp, _w_shift);
                    pp += 8;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128 _p0 = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), 0.f, 0.f);
                    __m128 _p1 = _mm_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), 0.f, 0.f);
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_unpacklo_epi8(_q0, _q1)));
                    pp += 4;
                    p0 += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), 0.f, 0.f);
                    unsigned int q = (unsigned int)float2int8_sse(_mm_mul_ps(_p, _scale));
                    pp[0] = (signed char)q;
                    pp[1] = (signed char)(q >> 8);
                    pp += 2;
                    p0++;
                }
#endif // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[A_hstep]);
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                    p0++;
                }

                pd += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                const unsigned short* p0a = p0;
                float absmax = 0.f;
                int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _absmax512 = _mm512_setzero_ps();
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a)));
                    _absmax512 = _mm512_max_ps(_absmax512, abs512_ps(_p));
                    p0a += 16;
                }
                absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax512));
#endif // __AVX512F__
                __m256 _absmax256 = _mm256_setzero_ps();
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a)));
                    _absmax256 = _mm256_max_ps(_absmax256, abs256_ps(_p));
                    p0a += 8;
                }
                absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax256));
#endif // __AVX__
                __m128 _absmax128 = _mm_setzero_ps();
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                    _absmax128 = _mm_max_ps(_absmax128, abs_ps(_p));
                    p0a += 4;
                }
                absmax = std::max(absmax, _mm_reduce_max_ps(_absmax128));
#endif // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0a++);
                    absmax = std::max(absmax, (float)fabsf(v));
                }

                if (absmax == 0.f)
                {
                    pd[0] = 0.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    memset(pp, 0, max_kk0 >= 4 ? max_kk0 + 4 : max_kk0);
                    pp += max_kk0 + (max_kk0 >= 4 ? 4 : 0);
#else
                    memset(pp, 0, max_kk0);
                    pp += max_kk0;
#endif
                    p0 += max_kk0;
                    pd++;
                    continue;
                }

                const float scale = 127.f / absmax;
                pd[0] = absmax / 127.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                int w_shift = 0;
#endif
                kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _scale512 = _mm512_set1_ps(scale);
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                    __m128i _q = float2int8_avx512(_mm512_mul_ps(_p, _scale512));
                    _mm_storeu_si128((__m128i*)pp, _q);
                    pp += 16;
                    p0 += 16;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m256i _q16 = _mm256_cvtepi8_epi16(_q);
                    __m256i _q32 = _mm256_madd_epi16(_q16, _mm256_set1_epi16(1));
                    w_shift += _mm_reduce_add_epi32(_mm256_castsi256_si128(_q32));
                    w_shift += _mm_reduce_add_epi32(_mm256_extracti128_si256(_q32, 1));
#endif
                }
#endif // __AVX512F__
                __m256 _scale256 = _mm256_set1_ps(scale);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                    const int64_t q = float2int8_avx(_mm256_mul_ps(_p, _scale256));
                    _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(q));
                    pp += 8;
                    p0 += 8;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
#if defined(__x86_64__) || defined(_M_X64)
                    __m128i _q8 = _mm_cvtsi64_si128(q);
#else
                    __m128i _q8 = _mm_loadl_epi64((const __m128i*)(pp - 8));
#endif
                    __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                    w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                }
#endif // __AVX__
                __m128 _scale128 = _mm_set1_ps(scale);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                    const int32_t q = float2int8_sse(_mm_mul_ps(_p, _scale128));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(q)));
                    pp += 4;
                    p0 += 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _q8 = _mm_cvtsi32_si128(q);
                    __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                    w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                }
#endif // __SSE2__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(w_shift * 127)));
                    pp += 4;
                }
#endif
                for (; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0++);
                    *pp++ = float2int8(v * scale);
                }

                pd++;
            }
        }
        return;
    }
    const float* input_scale_ptr = (const float*)input_scales + k;

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const int block_count = (max_kk + block_size - 1) / block_size;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

            if (elempack == 16)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p), _mm512_set1_ps(psa[0])));
                    p0a += 16;
                    psa++;
                }

                __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512 _p0 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _mm512_set1_ps(ps[0]));
                    __m512 _p1 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _mm512_set1_ps(ps[1]));
                    __m512 _p2 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 32))), _mm512_set1_ps(ps[2]));
                    __m512 _p3 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 48))), _mm512_set1_ps(ps[3]));
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_p2, _scale));
                    __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_p3, _scale));
#if __AVX512VNNI__
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                    _mm512_storeu_si512((__m512i*)pp, _q);
                    _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                    pp += 64;
                    p0 += 64;
                    ps += 4;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm512_storeu_si512((__m512i*)pp, _w_shift);
                    pp += 64;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m512 _p0 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _mm512_set1_ps(ps[0]));
                    __m512 _p1 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _mm512_set1_ps(ps[1]));
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                    pp += 32;
                    p0 += 32;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m512 _p = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _mm512_set1_ps(ps[0]));
                    _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                    pp += 16;
                    p0 += 16;
                    ps++;
                }

                pd += 16;
            }
            if (elempack == 8)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m512 _p;
                    _p = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a)), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 8))));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p), _mm512_set1_ps(psa[0])));
                    p0a += 8;
                    psa++;
                }
                __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512 _p0, _p1, _p2, _p3;

                    {
                        _p0 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 0 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 0 * 8))));
                        _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(ps[0]));
                    }
                    {
                        _p1 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 1 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 1 * 8))));
                        _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(ps[1]));
                    }
                    {
                        _p2 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 2 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 2 * 8))));
                        _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(ps[2]));
                    }
                    {
                        _p3 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 3 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 3 * 8))));
                        _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(ps[3]));
                    }
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_p2, _scale));
                    __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_p3, _scale));
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                    _mm512_storeu_si512((__m512i*)pp, _q);
                    _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                    pp += 64;
                    p0 += 8 * 4;
                    ps += 4;
                }
#endif
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm512_storeu_si512((__m512i*)pp, _w_shift);
                    pp += 64;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m512 _p0, _p1;

                    {
                        _p0 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 0 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 0 * 8))));
                        _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(ps[0]));
                    }
                    {
                        _p1 = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 1 * 8))), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 1 * 8))));
                        _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(ps[1]));
                    }
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                    pp += 32;
                    p0 += 8 * 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m512 _p = combine8x2_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8))));
                    _p = _mm512_mul_ps(_p, _mm512_set1_ps(ps[0]));
                    _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                    pp += 16;
                    p0 += 8;
                    ps++;
                }
                pd += 16;
            }
            if (elempack == 4)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m512 _p;
                    _p = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 8))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 12))));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p), _mm512_set1_ps(psa[0])));
                    p0a += 4;
                    psa++;
                }
                __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512 _p0, _p1, _p2, _p3;

                    {
                        _p0 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 0 * 4))));
                        _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(ps[0]));
                    }
                    {
                        _p1 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 1 * 4))));
                        _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(ps[1]));
                    }
                    {
                        _p2 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 2 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 2 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 2 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 2 * 4))));
                        _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(ps[2]));
                    }
                    {
                        _p3 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 3 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 3 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 3 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 3 * 4))));
                        _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(ps[3]));
                    }
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_p2, _scale));
                    __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_p3, _scale));
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                    _mm512_storeu_si512((__m512i*)pp, _q);
                    _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                    pp += 64;
                    p0 += 4 * 4;
                    ps += 4;
                }
#endif
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm512_storeu_si512((__m512i*)pp, _w_shift);
                    pp += 64;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m512 _p0, _p1;

                    {
                        _p0 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 0 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 0 * 4))));
                        _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(ps[0]));
                    }
                    {
                        _p1 = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8 + 1 * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12 + 1 * 4))));
                        _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(ps[1]));
                    }
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                    pp += 32;
                    p0 += 4 * 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m512 _p = combine4x4_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12))));
                    _p = _mm512_mul_ps(_p, _mm512_set1_ps(ps[0]));
                    _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                    pp += 16;
                    p0 += 4;
                    ps++;
                }
                pd += 16;
            }
            if (elempack == 1)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;

                __m512 _absmax0 = _mm512_setzero_ps();
                __m512 _absmax1 = _mm512_setzero_ps();
                __m512 _absmax2 = _mm512_setzero_ps();
                __m512 _absmax3 = _mm512_setzero_ps();
                __m512 _absmax4 = _mm512_setzero_ps();
                __m512 _absmax5 = _mm512_setzero_ps();
                __m512 _absmax6 = _mm512_setzero_ps();
                __m512 _absmax7 = _mm512_setzero_ps();
                __m512 _absmax8 = _mm512_setzero_ps();
                __m512 _absmax9 = _mm512_setzero_ps();
                __m512 _absmaxa = _mm512_setzero_ps();
                __m512 _absmaxb = _mm512_setzero_ps();
                __m512 _absmaxc = _mm512_setzero_ps();
                __m512 _absmaxd = _mm512_setzero_ps();
                __m512 _absmaxe = _mm512_setzero_ps();
                __m512 _absmaxf = _mm512_setzero_ps();
                int kk = 0;
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    __m512 _s = _mm512_loadu_ps(psa);
                    __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a)));
                    _absmax0 = _mm512_max_ps(_absmax0, _mm512_mul_ps(abs512_ps(_p0), _s));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep)));
                    _absmax1 = _mm512_max_ps(_absmax1, _mm512_mul_ps(abs512_ps(_p1), _s));
                    __m512 _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 2)));
                    _absmax2 = _mm512_max_ps(_absmax2, _mm512_mul_ps(abs512_ps(_p2), _s));
                    __m512 _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 3)));
                    _absmax3 = _mm512_max_ps(_absmax3, _mm512_mul_ps(abs512_ps(_p3), _s));
                    __m512 _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 4)));
                    _absmax4 = _mm512_max_ps(_absmax4, _mm512_mul_ps(abs512_ps(_p4), _s));
                    __m512 _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 5)));
                    _absmax5 = _mm512_max_ps(_absmax5, _mm512_mul_ps(abs512_ps(_p5), _s));
                    __m512 _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 6)));
                    _absmax6 = _mm512_max_ps(_absmax6, _mm512_mul_ps(abs512_ps(_p6), _s));
                    __m512 _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 7)));
                    _absmax7 = _mm512_max_ps(_absmax7, _mm512_mul_ps(abs512_ps(_p7), _s));
                    __m512 _p8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 8)));
                    _absmax8 = _mm512_max_ps(_absmax8, _mm512_mul_ps(abs512_ps(_p8), _s));
                    __m512 _p9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 9)));
                    _absmax9 = _mm512_max_ps(_absmax9, _mm512_mul_ps(abs512_ps(_p9), _s));
                    __m512 _pa = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 10)));
                    _absmaxa = _mm512_max_ps(_absmaxa, _mm512_mul_ps(abs512_ps(_pa), _s));
                    __m512 _pb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 11)));
                    _absmaxb = _mm512_max_ps(_absmaxb, _mm512_mul_ps(abs512_ps(_pb), _s));
                    __m512 _pc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 12)));
                    _absmaxc = _mm512_max_ps(_absmaxc, _mm512_mul_ps(abs512_ps(_pc), _s));
                    __m512 _pd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 13)));
                    _absmaxd = _mm512_max_ps(_absmaxd, _mm512_mul_ps(abs512_ps(_pd), _s));
                    __m512 _pe = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 14)));
                    _absmaxe = _mm512_max_ps(_absmaxe, _mm512_mul_ps(abs512_ps(_pe), _s));
                    __m512 _pf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + A_hstep * 15)));
                    _absmaxf = _mm512_max_ps(_absmaxf, _mm512_mul_ps(abs512_ps(_pf), _s));
                    p0a += 16;
                    psa += 16;
                }

                float absmax0 = _mm512_reduce_max_ps(_absmax0);
                float absmax1 = _mm512_reduce_max_ps(_absmax1);
                float absmax2 = _mm512_reduce_max_ps(_absmax2);
                float absmax3 = _mm512_reduce_max_ps(_absmax3);
                float absmax4 = _mm512_reduce_max_ps(_absmax4);
                float absmax5 = _mm512_reduce_max_ps(_absmax5);
                float absmax6 = _mm512_reduce_max_ps(_absmax6);
                float absmax7 = _mm512_reduce_max_ps(_absmax7);
                float absmax8 = _mm512_reduce_max_ps(_absmax8);
                float absmax9 = _mm512_reduce_max_ps(_absmax9);
                float absmaxa = _mm512_reduce_max_ps(_absmaxa);
                float absmaxb = _mm512_reduce_max_ps(_absmaxb);
                float absmaxc = _mm512_reduce_max_ps(_absmaxc);
                float absmaxd = _mm512_reduce_max_ps(_absmaxd);
                float absmaxe = _mm512_reduce_max_ps(_absmaxe);
                float absmaxf = _mm512_reduce_max_ps(_absmaxf);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s = _mm_loadu_ps(psa);
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                    absmax0 = std::max(absmax0, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p0, _s))));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep)));
                    absmax1 = std::max(absmax1, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p1, _s))));
                    __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 2)));
                    absmax2 = std::max(absmax2, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p2, _s))));
                    __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 3)));
                    absmax3 = std::max(absmax3, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p3, _s))));
                    __m128 _p4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 4)));
                    absmax4 = std::max(absmax4, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p4, _s))));
                    __m128 _p5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 5)));
                    absmax5 = std::max(absmax5, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p5, _s))));
                    __m128 _p6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 6)));
                    absmax6 = std::max(absmax6, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p6, _s))));
                    __m128 _p7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 7)));
                    absmax7 = std::max(absmax7, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p7, _s))));
                    __m128 _p8 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 8)));
                    absmax8 = std::max(absmax8, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p8, _s))));
                    __m128 _p9 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 9)));
                    absmax9 = std::max(absmax9, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_p9, _s))));
                    __m128 _pa = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 10)));
                    absmaxa = std::max(absmaxa, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_pa, _s))));
                    __m128 _pb = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 11)));
                    absmaxb = std::max(absmaxb, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_pb, _s))));
                    __m128 _pc = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 12)));
                    absmaxc = std::max(absmaxc, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_pc, _s))));
                    __m128 _pd = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 13)));
                    absmaxd = std::max(absmaxd, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_pd, _s))));
                    __m128 _pe = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 14)));
                    absmaxe = std::max(absmaxe, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_pe, _s))));
                    __m128 _pf = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 15)));
                    absmaxf = std::max(absmaxf, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_pf, _s))));
                    p0a += 4;
                    psa += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])) * s);
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[A_hstep])) * s);
                    absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(p0a[A_hstep * 2])) * s);
                    absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(p0a[A_hstep * 3])) * s);
                    absmax4 = std::max(absmax4, fabsf(bfloat16_to_float32(p0a[A_hstep * 4])) * s);
                    absmax5 = std::max(absmax5, fabsf(bfloat16_to_float32(p0a[A_hstep * 5])) * s);
                    absmax6 = std::max(absmax6, fabsf(bfloat16_to_float32(p0a[A_hstep * 6])) * s);
                    absmax7 = std::max(absmax7, fabsf(bfloat16_to_float32(p0a[A_hstep * 7])) * s);
                    absmax8 = std::max(absmax8, fabsf(bfloat16_to_float32(p0a[A_hstep * 8])) * s);
                    absmax9 = std::max(absmax9, fabsf(bfloat16_to_float32(p0a[A_hstep * 9])) * s);
                    absmaxa = std::max(absmaxa, fabsf(bfloat16_to_float32(p0a[A_hstep * 10])) * s);
                    absmaxb = std::max(absmaxb, fabsf(bfloat16_to_float32(p0a[A_hstep * 11])) * s);
                    absmaxc = std::max(absmaxc, fabsf(bfloat16_to_float32(p0a[A_hstep * 12])) * s);
                    absmaxd = std::max(absmaxd, fabsf(bfloat16_to_float32(p0a[A_hstep * 13])) * s);
                    absmaxe = std::max(absmaxe, fabsf(bfloat16_to_float32(p0a[A_hstep * 14])) * s);
                    absmaxf = std::max(absmaxf, fabsf(bfloat16_to_float32(p0a[A_hstep * 15])) * s);
                    p0a++;
                }
                __m512 _absmax = _mm512_setr_ps(absmax0, absmax1, absmax2, absmax3, absmax4, absmax5, absmax6, absmax7, absmax8, absmax9, absmaxa, absmaxb, absmaxc, absmaxd, absmaxe, absmaxf);

                __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                _mm512_storeu_ps(pd, _descale);

#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                kk = 0;
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    {
                        __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep)));
                        __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2)));
                        __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3)));
                        __m256 _p4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4)));
                        __m256 _p5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 5)));
                        __m256 _p6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 6)));
                        __m256 _p7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 7)));
                        __m256 _p8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8)));
                        __m256 _p9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 9)));
                        __m256 _pa = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 10)));
                        __m256 _pb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 11)));
                        __m256 _pc = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 12)));
                        __m256 _pd = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 13)));
                        __m256 _pe = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 14)));
                        __m256 _pf = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 15)));
                        __m256 _s = _mm256_loadu_ps(ps);
                        _p0 = _mm256_mul_ps(_p0, _s);
                        _p1 = _mm256_mul_ps(_p1, _s);
                        _p2 = _mm256_mul_ps(_p2, _s);
                        _p3 = _mm256_mul_ps(_p3, _s);
                        _p4 = _mm256_mul_ps(_p4, _s);
                        _p5 = _mm256_mul_ps(_p5, _s);
                        _p6 = _mm256_mul_ps(_p6, _s);
                        _p7 = _mm256_mul_ps(_p7, _s);
                        _p8 = _mm256_mul_ps(_p8, _s);
                        _p9 = _mm256_mul_ps(_p9, _s);
                        _pa = _mm256_mul_ps(_pa, _s);
                        _pb = _mm256_mul_ps(_pb, _s);
                        _pc = _mm256_mul_ps(_pc, _s);
                        _pd = _mm256_mul_ps(_pd, _s);
                        _pe = _mm256_mul_ps(_pe, _s);
                        _pf = _mm256_mul_ps(_pf, _s);
                        transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                        transpose8x8_ps(_p8, _p9, _pa, _pb, _pc, _pd, _pe, _pf);

                        __m512 _t0 = combine8x2_ps(_p0, _p8);
                        __m512 _t1 = combine8x2_ps(_p1, _p9);
                        __m512 _t2 = combine8x2_ps(_p2, _pa);
                        __m512 _t3 = combine8x2_ps(_p3, _pb);
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                        _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                        pp += 64;

                        _t0 = combine8x2_ps(_p4, _pc);
                        _t1 = combine8x2_ps(_p5, _pd);
                        _t2 = combine8x2_ps(_p6, _pe);
                        _t3 = combine8x2_ps(_p7, _pf);
                        _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                        _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                        _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                        _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                        _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                        pp += 64;
                    }
                    {
                        __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep + 8)));
                        __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2 + 8)));
                        __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3 + 8)));
                        __m256 _p4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4 + 8)));
                        __m256 _p5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 5 + 8)));
                        __m256 _p6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 6 + 8)));
                        __m256 _p7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 7 + 8)));
                        __m256 _p8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 8 + 8)));
                        __m256 _p9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 9 + 8)));
                        __m256 _pa = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 10 + 8)));
                        __m256 _pb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 11 + 8)));
                        __m256 _pc = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 12 + 8)));
                        __m256 _pd = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 13 + 8)));
                        __m256 _pe = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 14 + 8)));
                        __m256 _pf = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 15 + 8)));
                        __m256 _s = _mm256_loadu_ps(ps + 8);
                        _p0 = _mm256_mul_ps(_p0, _s);
                        _p1 = _mm256_mul_ps(_p1, _s);
                        _p2 = _mm256_mul_ps(_p2, _s);
                        _p3 = _mm256_mul_ps(_p3, _s);
                        _p4 = _mm256_mul_ps(_p4, _s);
                        _p5 = _mm256_mul_ps(_p5, _s);
                        _p6 = _mm256_mul_ps(_p6, _s);
                        _p7 = _mm256_mul_ps(_p7, _s);
                        _p8 = _mm256_mul_ps(_p8, _s);
                        _p9 = _mm256_mul_ps(_p9, _s);
                        _pa = _mm256_mul_ps(_pa, _s);
                        _pb = _mm256_mul_ps(_pb, _s);
                        _pc = _mm256_mul_ps(_pc, _s);
                        _pd = _mm256_mul_ps(_pd, _s);
                        _pe = _mm256_mul_ps(_pe, _s);
                        _pf = _mm256_mul_ps(_pf, _s);
                        transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                        transpose8x8_ps(_p8, _p9, _pa, _pb, _pc, _pd, _pe, _pf);

                        __m512 _t0 = combine8x2_ps(_p0, _p8);
                        __m512 _t1 = combine8x2_ps(_p1, _p9);
                        __m512 _t2 = combine8x2_ps(_p2, _pa);
                        __m512 _t3 = combine8x2_ps(_p3, _pb);
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                        _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                        pp += 64;

                        _t0 = combine8x2_ps(_p4, _pc);
                        _t1 = combine8x2_ps(_p5, _pd);
                        _t2 = combine8x2_ps(_p6, _pe);
                        _t3 = combine8x2_ps(_p7, _pf);
                        _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                        _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                        _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                        _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                        _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                        pp += 64;
                    }
                    p0 += 16;
                    ps += 16;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                    __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                    __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                    __m128 _p4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4)));
                    __m128 _p5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 5)));
                    __m128 _p6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 6)));
                    __m128 _p7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 7)));
                    __m128 _p8 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 8)));
                    __m128 _p9 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 9)));
                    __m128 _pa = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 10)));
                    __m128 _pb = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 11)));
                    __m128 _pc = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 12)));
                    __m128 _pd = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 13)));
                    __m128 _pe = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 14)));
                    __m128 _pf = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 15)));
                    __m128 _s = _mm_loadu_ps(ps);
                    _p0 = _mm_mul_ps(_p0, _s);
                    _p1 = _mm_mul_ps(_p1, _s);
                    _p2 = _mm_mul_ps(_p2, _s);
                    _p3 = _mm_mul_ps(_p3, _s);
                    _p4 = _mm_mul_ps(_p4, _s);
                    _p5 = _mm_mul_ps(_p5, _s);
                    _p6 = _mm_mul_ps(_p6, _s);
                    _p7 = _mm_mul_ps(_p7, _s);
                    _p8 = _mm_mul_ps(_p8, _s);
                    _p9 = _mm_mul_ps(_p9, _s);
                    _pa = _mm_mul_ps(_pa, _s);
                    _pb = _mm_mul_ps(_pb, _s);
                    _pc = _mm_mul_ps(_pc, _s);
                    _pd = _mm_mul_ps(_pd, _s);
                    _pe = _mm_mul_ps(_pe, _s);
                    _pf = _mm_mul_ps(_pf, _s);
                    __m512 _t0 = combine4x4_ps(_p0, _p4, _p8, _pc);
                    __m512 _t1 = combine4x4_ps(_p1, _p5, _p9, _pd);
                    __m512 _t2 = combine4x4_ps(_p2, _p6, _pa, _pe);
                    __m512 _t3 = combine4x4_ps(_p3, _p7, _pb, _pf);
                    __m512 _t4 = _mm512_unpacklo_ps(_t0, _t1);
                    __m512 _t5 = _mm512_unpackhi_ps(_t0, _t1);
                    __m512 _t6 = _mm512_unpacklo_ps(_t2, _t3);
                    __m512 _t7 = _mm512_unpackhi_ps(_t2, _t3);
                    _t0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_t4), _mm512_castps_pd(_t6)));
                    _t1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_t4), _mm512_castps_pd(_t6)));
                    _t2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_t5), _mm512_castps_pd(_t7)));
                    _t3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_t5), _mm512_castps_pd(_t7)));
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_t0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_t1, _scale));
                    __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_t2, _scale));
                    __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_t3, _scale));
#if __AVX512VNNI__
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                    _mm512_storeu_si512((__m512i*)pp, _q);
                    _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi8(_q2, _q3));
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpackhi_epi8(_q2, _q3));
#endif
                    pp += 64;
                    p0 += 4;
                    ps += 4;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm512_storeu_si512((__m512i*)pp, _w_shift);
                    pp += 64;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m512 _p0 = _mm512_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]), bfloat16_to_float32(p0[A_hstep * 8]), bfloat16_to_float32(p0[A_hstep * 9]), bfloat16_to_float32(p0[A_hstep * 10]), bfloat16_to_float32(p0[A_hstep * 11]), bfloat16_to_float32(p0[A_hstep * 12]), bfloat16_to_float32(p0[A_hstep * 13]), bfloat16_to_float32(p0[A_hstep * 14]), bfloat16_to_float32(p0[A_hstep * 15]));
                    __m512 _p1 = _mm512_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), bfloat16_to_float32(p0[A_hstep * 2 + 1]), bfloat16_to_float32(p0[A_hstep * 3 + 1]), bfloat16_to_float32(p0[A_hstep * 4 + 1]), bfloat16_to_float32(p0[A_hstep * 5 + 1]), bfloat16_to_float32(p0[A_hstep * 6 + 1]), bfloat16_to_float32(p0[A_hstep * 7 + 1]), bfloat16_to_float32(p0[A_hstep * 8 + 1]), bfloat16_to_float32(p0[A_hstep * 9 + 1]), bfloat16_to_float32(p0[A_hstep * 10 + 1]), bfloat16_to_float32(p0[A_hstep * 11 + 1]), bfloat16_to_float32(p0[A_hstep * 12 + 1]), bfloat16_to_float32(p0[A_hstep * 13 + 1]), bfloat16_to_float32(p0[A_hstep * 14 + 1]), bfloat16_to_float32(p0[A_hstep * 15 + 1]));
                    _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(ps[0]));
                    _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(ps[1]));
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                    pp += 32;
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m512 _p = _mm512_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]), bfloat16_to_float32(p0[A_hstep * 8]), bfloat16_to_float32(p0[A_hstep * 9]), bfloat16_to_float32(p0[A_hstep * 10]), bfloat16_to_float32(p0[A_hstep * 11]), bfloat16_to_float32(p0[A_hstep * 12]), bfloat16_to_float32(p0[A_hstep * 13]), bfloat16_to_float32(p0[A_hstep * 14]), bfloat16_to_float32(p0[A_hstep * 15]));
                    _p = _mm512_mul_ps(_p, _mm512_set1_ps(ps[0]));
                    _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                    pp += 16;
                    p0++;
                    ps++;
                }

                pd += 16;
            }
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    signed char* pp1 = pp + AT_tile.w * 4;
    float* pd1 = pd + AT_descales_tile.w * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

            if (elempack == 8)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m256 _absmax = _mm256_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a));
                    _absmax = _mm256_max_ps(_absmax, _mm256_mul_ps(abs256_ps(_p), _mm256_set1_ps(psa[0])));
                    p0a += 8;
                    psa++;
                }

                __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                _mm256_storeu_ps(pd, _descale);
#else
                _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _w_shift = _mm256_setzero_si256();
#endif
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256 _p0 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _mm256_set1_ps(ps[0]));
                    __m256 _p1 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _mm256_set1_ps(ps[1]));
                    __m256 _p2 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 16))), _mm256_set1_ps(ps[2]));
                    __m256 _p3 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 24))), _mm256_set1_ps(ps[3]));
                    _p0 = _mm256_mul_ps(_p0, _scale);
                    _p1 = _mm256_mul_ps(_p1, _scale);
                    _p2 = _mm256_mul_ps(_p2, _scale);
                    _p3 = _mm256_mul_ps(_p3, _scale);
                    __m128i _q0 = float2int8_avx(_p0, _p2);
                    __m128i _q1 = float2int8_avx(_p1, _p3);
                    __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                    _q0 = _mm_unpacklo_epi16(_q01, _q23);
                    _q1 = _mm_unpackhi_epi16(_q01, _q23);
                    __m256i _q = combine4x2_epi32(_q0, _q1);
                    _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q01);
                    _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                    _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif
#if __AVX2__
                    pp += 32;
#else
                    pp += 16;
                    pp1 += 16;
#endif
                    p0 += 32;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm256_storeu_si256((__m256i*)pp, _w_shift);
                    pp += 32;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256 _p0 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _mm256_set1_ps(ps[0]));
                    __m256 _p1 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _mm256_set1_ps(ps[1]));
                    __m128i _q = float2int8_avx(_mm256_mul_ps(_p0, _scale), _mm256_mul_ps(_p1, _scale));
                    __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                    _q = _mm_shuffle_epi8(_q, _si);
#if __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q);
                    pp += 16;
#else
                    _mm_storel_pd((double*)pp, _mm_castsi128_pd(_q));
                    _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q));
                    pp += 8;
                    pp1 += 8;
#endif
                    p0 += 16;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m256 _p = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _mm256_set1_ps(ps[0]));
                    _p = _mm256_mul_ps(_p, _scale);
#if __AVX2__
                    _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(float2int8_avx(_p)));
                    pp += 8;
#else
                    const uint64_t q = (uint64_t)float2int8_avx(_p);
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128((int)q)));
                    _mm_store_ss((float*)pp1, _mm_castsi128_ps(_mm_cvtsi32_si128((int)(q >> 32))));
                    pp += 4;
                    pp1 += 4;
#endif
                    p0 += 8;
                    ps++;
                }

#if __AVX2__
                pd += 8;
#else
                pd += 4;
                pd1 += 4;
#endif
            }
            if (elempack == 4)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m256 _absmax = _mm256_setzero_ps();
                int kk = 0;
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 4)));
                    __m256 _t0 = _mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _t1 = _mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 3, 0, 1));
                    _absmax = _mm256_max_ps(_absmax, _mm256_mul_ps(abs256_ps(_t0), _mm256_set1_ps(psa[0])));
                    _absmax = _mm256_max_ps(_absmax, _mm256_mul_ps(abs256_ps(_t1), _mm256_set1_ps(psa[1])));
                    p0a += 8;
                    psa += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m256 _p = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 4))));
                    _absmax = _mm256_max_ps(_absmax, _mm256_mul_ps(abs256_ps(_p), _mm256_set1_ps(psa[0])));
                    p0a += 4;
                    psa++;
                }

                __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                _mm256_storeu_ps(pd, _descale);
#else
                _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _w_shift = _mm256_setzero_si256();
#endif
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
                    __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4)));
                    __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4 + 8)));
                    __m256 _t0 = _mm256_mul_ps(_mm256_permute2f128_ps(_p0, _p2, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_set1_ps(ps[0]));
                    __m256 _t1 = _mm256_mul_ps(_mm256_permute2f128_ps(_p0, _p2, _MM_SHUFFLE(0, 3, 0, 1)), _mm256_set1_ps(ps[1]));
                    __m256 _t2 = _mm256_mul_ps(_mm256_permute2f128_ps(_p1, _p3, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_set1_ps(ps[2]));
                    __m256 _t3 = _mm256_mul_ps(_mm256_permute2f128_ps(_p1, _p3, _MM_SHUFFLE(0, 3, 0, 1)), _mm256_set1_ps(ps[3]));
                    _t0 = _mm256_mul_ps(_t0, _scale);
                    _t1 = _mm256_mul_ps(_t1, _scale);
                    _t2 = _mm256_mul_ps(_t2, _scale);
                    _t3 = _mm256_mul_ps(_t3, _scale);
                    __m128i _q0 = float2int8_avx(_t0, _t2);
                    __m128i _q1 = float2int8_avx(_t1, _t3);
                    __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                    _q0 = _mm_unpacklo_epi16(_q01, _q23);
                    _q1 = _mm_unpackhi_epi16(_q01, _q23);
                    __m256i _q = combine4x2_epi32(_q0, _q1);
                    _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q01);
                    _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                    _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif
#if __AVX2__
                    pp += 32;
#else
                    pp += 16;
                    pp1 += 16;
#endif
                    p0 += 16;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm256_storeu_si256((__m256i*)pp, _w_shift);
                    pp += 32;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4)));
                    __m256 _t0 = _mm256_mul_ps(_mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_set1_ps(ps[0]));
                    __m256 _t1 = _mm256_mul_ps(_mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 3, 0, 1)), _mm256_set1_ps(ps[1]));
                    __m128i _q = float2int8_avx(_mm256_mul_ps(_t0, _scale), _mm256_mul_ps(_t1, _scale));
                    __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                    _q = _mm_shuffle_epi8(_q, _si);
#if __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q);
                    pp += 16;
#else
                    _mm_storel_pd((double*)pp, _mm_castsi128_pd(_q));
                    _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q));
                    pp += 8;
                    pp1 += 8;
#endif
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m256 _p = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4))));
                    _p = _mm256_mul_ps(_mm256_mul_ps(_p, _mm256_set1_ps(ps[0])), _scale);
#if __AVX2__
                    _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(float2int8_avx(_p)));
                    pp += 8;
#else
                    const uint64_t q = (uint64_t)float2int8_avx(_p);
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128((int)q)));
                    _mm_store_ss((float*)pp1, _mm_castsi128_ps(_mm_cvtsi32_si128((int)(q >> 32))));
                    pp += 4;
                    pp1 += 4;
#endif
                    p0 += 4;
                    ps++;
                }

#if __AVX2__
                pd += 8;
#else
                pd += 4;
                pd1 += 4;
#endif
            }
            if (elempack == 1)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;

                __m256 _absmax0 = _mm256_setzero_ps();
                __m256 _absmax1 = _mm256_setzero_ps();
                __m256 _absmax2 = _mm256_setzero_ps();
                __m256 _absmax3 = _mm256_setzero_ps();
                __m256 _absmax4 = _mm256_setzero_ps();
                __m256 _absmax5 = _mm256_setzero_ps();
                __m256 _absmax6 = _mm256_setzero_ps();
                __m256 _absmax7 = _mm256_setzero_ps();
                int kk = 0;
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _s = _mm256_loadu_ps(psa);
                    __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a)));
                    _absmax0 = _mm256_max_ps(_absmax0, _mm256_mul_ps(abs256_ps(_p0), _s));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep)));
                    _absmax1 = _mm256_max_ps(_absmax1, _mm256_mul_ps(abs256_ps(_p1), _s));
                    __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 2)));
                    _absmax2 = _mm256_max_ps(_absmax2, _mm256_mul_ps(abs256_ps(_p2), _s));
                    __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 3)));
                    _absmax3 = _mm256_max_ps(_absmax3, _mm256_mul_ps(abs256_ps(_p3), _s));
                    __m256 _p4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 4)));
                    _absmax4 = _mm256_max_ps(_absmax4, _mm256_mul_ps(abs256_ps(_p4), _s));
                    __m256 _p5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 5)));
                    _absmax5 = _mm256_max_ps(_absmax5, _mm256_mul_ps(abs256_ps(_p5), _s));
                    __m256 _p6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 6)));
                    _absmax6 = _mm256_max_ps(_absmax6, _mm256_mul_ps(abs256_ps(_p6), _s));
                    __m256 _p7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + A_hstep * 7)));
                    _absmax7 = _mm256_max_ps(_absmax7, _mm256_mul_ps(abs256_ps(_p7), _s));
                    p0a += 8;
                    psa += 8;
                }

                float absmax0 = _mm256_reduce_max_ps(_absmax0);
                float absmax1 = _mm256_reduce_max_ps(_absmax1);
                float absmax2 = _mm256_reduce_max_ps(_absmax2);
                float absmax3 = _mm256_reduce_max_ps(_absmax3);
                float absmax4 = _mm256_reduce_max_ps(_absmax4);
                float absmax5 = _mm256_reduce_max_ps(_absmax5);
                float absmax6 = _mm256_reduce_max_ps(_absmax6);
                float absmax7 = _mm256_reduce_max_ps(_absmax7);
                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])) * s);
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[A_hstep])) * s);
                    absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(p0a[A_hstep * 2])) * s);
                    absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(p0a[A_hstep * 3])) * s);
                    absmax4 = std::max(absmax4, fabsf(bfloat16_to_float32(p0a[A_hstep * 4])) * s);
                    absmax5 = std::max(absmax5, fabsf(bfloat16_to_float32(p0a[A_hstep * 5])) * s);
                    absmax6 = std::max(absmax6, fabsf(bfloat16_to_float32(p0a[A_hstep * 6])) * s);
                    absmax7 = std::max(absmax7, fabsf(bfloat16_to_float32(p0a[A_hstep * 7])) * s);
                    p0a++;
                }

                __m256 _absmax = _mm256_setr_ps(absmax0, absmax1, absmax2, absmax3, absmax4, absmax5, absmax6, absmax7);
                __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                _mm256_storeu_ps(pd, _descale);
#else
                _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _w_shift = _mm256_setzero_si256();
#endif
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                    __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                    __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                    __m128 _p4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 4)));
                    __m128 _p5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 5)));
                    __m128 _p6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 6)));
                    __m128 _p7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 7)));
                    __m128 _s = _mm_loadu_ps(ps);
                    _p0 = _mm_mul_ps(_p0, _s);
                    _p1 = _mm_mul_ps(_p1, _s);
                    _p2 = _mm_mul_ps(_p2, _s);
                    _p3 = _mm_mul_ps(_p3, _s);
                    _p4 = _mm_mul_ps(_p4, _s);
                    _p5 = _mm_mul_ps(_p5, _s);
                    _p6 = _mm_mul_ps(_p6, _s);
                    _p7 = _mm_mul_ps(_p7, _s);

                    __m256 _t0 = combine4x2_ps(_p0, _p4);
                    __m256 _t1 = combine4x2_ps(_p1, _p5);
                    __m256 _t2 = combine4x2_ps(_p2, _p6);
                    __m256 _t3 = combine4x2_ps(_p3, _p7);
                    __m256 _t4 = _mm256_unpacklo_ps(_t0, _t1);
                    __m256 _t5 = _mm256_unpackhi_ps(_t0, _t1);
                    __m256 _t6 = _mm256_unpacklo_ps(_t2, _t3);
                    __m256 _t7 = _mm256_unpackhi_ps(_t2, _t3);
                    _t0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_t4), _mm256_castps_pd(_t6)));
                    _t1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_t4), _mm256_castps_pd(_t6)));
                    _t2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_t5), _mm256_castps_pd(_t7)));
                    _t3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_t5), _mm256_castps_pd(_t7)));
                    _t0 = _mm256_mul_ps(_t0, _scale);
                    _t1 = _mm256_mul_ps(_t1, _scale);
                    _t2 = _mm256_mul_ps(_t2, _scale);
                    _t3 = _mm256_mul_ps(_t3, _scale);

                    __m128i _q0 = float2int8_avx(_t0, _t2);
                    __m128i _q1 = float2int8_avx(_t1, _t3);
                    __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                    _q0 = _mm_unpacklo_epi16(_q01, _q23);
                    _q1 = _mm_unpackhi_epi16(_q01, _q23);
                    __m256i _q = combine4x2_epi32(_q0, _q1);
                    _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q01);
                    _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                    _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX2__
                    pp += 32;
#else
                    pp += 16;
                    pp1 += 16;
#endif
                    p0 += 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm256_storeu_si256((__m256i*)pp, _w_shift);
                    pp += 32;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __AVX2__
                    __m256 _p0 = _mm256_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
                    __m256 _p1 = _mm256_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), bfloat16_to_float32(p0[A_hstep * 2 + 1]), bfloat16_to_float32(p0[A_hstep * 3 + 1]), bfloat16_to_float32(p0[A_hstep * 4 + 1]), bfloat16_to_float32(p0[A_hstep * 5 + 1]), bfloat16_to_float32(p0[A_hstep * 6 + 1]), bfloat16_to_float32(p0[A_hstep * 7 + 1]));
#else
                    __m128 _p00 = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                    __m128 _p01 = _mm_setr_ps(bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
                    __m128 _p10 = _mm_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), bfloat16_to_float32(p0[A_hstep * 2 + 1]), bfloat16_to_float32(p0[A_hstep * 3 + 1]));
                    __m128 _p11 = _mm_setr_ps(bfloat16_to_float32(p0[A_hstep * 4 + 1]), bfloat16_to_float32(p0[A_hstep * 5 + 1]), bfloat16_to_float32(p0[A_hstep * 6 + 1]), bfloat16_to_float32(p0[A_hstep * 7 + 1]));
                    __m256 _p0 = combine4x2_ps(_p00, _p01);
                    __m256 _p1 = combine4x2_ps(_p10, _p11);
#endif // __AVX2__
                    _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(ps[0]));
                    _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(ps[1]));
                    _p0 = _mm256_mul_ps(_p0, _scale);
                    _p1 = _mm256_mul_ps(_p1, _scale);
                    __m128i _q = float2int8_avx(_p0, _p1);
                    __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                    _q = _mm_shuffle_epi8(_q, _si);
#if __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q);
                    pp += 16;
#else
                    _mm_storel_pd((double*)pp, _mm_castsi128_pd(_q));
                    _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q));
                    pp += 8;
                    pp1 += 8;
#endif
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
#if __AVX2__
                    __m256 _p = _mm256_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
#else
                    __m128 _p0 = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                    __m128 _p1 = _mm_setr_ps(bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
                    __m256 _p = combine4x2_ps(_p0, _p1);
#endif // __AVX2__
                    _p = _mm256_mul_ps(_p, _mm256_set1_ps(ps[0]));
#if __AVX2__
                    _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(float2int8_avx(_mm256_mul_ps(_p, _scale))));
                    pp += 8;
#else
                    const uint64_t q = (uint64_t)float2int8_avx(_mm256_mul_ps(_p, _scale));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128((int)q)));
                    _mm_store_ss((float*)pp1, _mm_castsi128_ps(_mm_cvtsi32_si128((int)(q >> 32))));
                    pp += 4;
                    pp1 += 4;
#endif
                    p0++;
                    ps++;
                }

#if __AVX2__
                pd += 8;
#else
                pd += 4;
                pd1 += 4;
#endif
            }
        }
#if !__AVX2__
        pp = pp1;
        pp1 = pp + AT_tile.w * 4;
        pd = pd1;
        pd1 = pd + AT_descales_tile.w * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

            if (elempack == 4)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m128 _absmax = _mm_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a));
                    _absmax = _mm_max_ps(_absmax, _mm_mul_ps(abs_ps(_p), _mm_set1_ps(psa[0])));
                    p0a += 4;
                    psa++;
                }
                __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _w_shift = _mm_setzero_si128();
#endif
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), _mm_set1_ps(ps[0]));
                    __m128 _p1 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))), _mm_set1_ps(ps[1]));
                    __m128 _p2 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 8))), _mm_set1_ps(ps[2]));
                    __m128 _p3 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 12))), _mm_set1_ps(ps[3]));
                    __m128i _q = float2int8_sse(_mm_mul_ps(_p0, _scale), _mm_mul_ps(_p1, _scale), _mm_mul_ps(_p2, _scale), _mm_mul_ps(_p3, _scale));
                    _q = _mm_shuffle_epi8(_q, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
                    _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
                    pp += 16;
                    p0 += 16;
                    ps += 4;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_storeu_si128((__m128i*)pp, _w_shift);
                    pp += 16;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128 _p0 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), _mm_set1_ps(ps[0]));
                    __m128 _p1 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))), _mm_set1_ps(ps[1]));
                    _p0 = _mm_mul_ps(_p0, _scale);
                    _p1 = _mm_mul_ps(_p1, _scale);
                    __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                    __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                    *(int64_t*)pp = float2int8_sse(_t0, _t1);
                    pp += 8;
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), _mm_set1_ps(ps[0]));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p, _scale)))));
                    pp += 4;
                    p0 += 4;
                    ps++;
                }
                pd += 4;
            }
            if (elempack == 1)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;

                __m128 _absmax0 = _mm_setzero_ps();
                __m128 _absmax1 = _mm_setzero_ps();
                __m128 _absmax2 = _mm_setzero_ps();
                __m128 _absmax3 = _mm_setzero_ps();
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s = _mm_loadu_ps(psa);
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                    _absmax0 = _mm_max_ps(_absmax0, _mm_mul_ps(abs_ps(_p0), _s));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep)));
                    _absmax1 = _mm_max_ps(_absmax1, _mm_mul_ps(abs_ps(_p1), _s));
                    __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 2)));
                    _absmax2 = _mm_max_ps(_absmax2, _mm_mul_ps(abs_ps(_p2), _s));
                    __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep * 3)));
                    _absmax3 = _mm_max_ps(_absmax3, _mm_mul_ps(abs_ps(_p3), _s));
                    p0a += 4;
                    psa += 4;
                }

                float absmax0 = _mm_reduce_max_ps(_absmax0);
                float absmax1 = _mm_reduce_max_ps(_absmax1);
                float absmax2 = _mm_reduce_max_ps(_absmax2);
                float absmax3 = _mm_reduce_max_ps(_absmax3);
                for (; kk < max_kk0; kk++)
                {
                    const float s = psa[0];
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])) * s);
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[A_hstep])) * s);
                    absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(p0a[A_hstep * 2])) * s);
                    absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(p0a[A_hstep * 3])) * s);
                    p0a++;
                    psa++;
                }

                __m128 _absmax = _mm_setr_ps(absmax0, absmax1, absmax2, absmax3);
                __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _w_shift = _mm_setzero_si128();
#endif
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                    __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                    __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                    __m128 _s = _mm_loadu_ps(ps);
                    _p0 = _mm_mul_ps(_p0, _s);
                    _p1 = _mm_mul_ps(_p1, _s);
                    _p2 = _mm_mul_ps(_p2, _s);
                    _p3 = _mm_mul_ps(_p3, _s);
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(0, 0, 0, 0)))));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(1, 1, 1, 1)))));
                    __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p2, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(2, 2, 2, 2)))));
                    __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p3, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(3, 3, 3, 3)))));
#if __AVX512VNNI__ || __AVXVNNI__
                    __m128i _q = _mm_unpacklo_epi64(_mm_unpacklo_epi32(_q0, _q1), _mm_unpacklo_epi32(_q2, _q3));
                    _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                    __m128i _q01 = _mm_unpacklo_epi16(_q0, _q1);
                    __m128i _q23 = _mm_unpacklo_epi16(_q2, _q3);
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi32(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
                    pp += 16;
                    p0 += 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_storeu_si128((__m128i*)pp, _w_shift);
                    pp += 16;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128 _p0 = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                    __m128 _p1 = _mm_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), bfloat16_to_float32(p0[A_hstep * 2 + 1]), bfloat16_to_float32(p0[A_hstep * 3 + 1]));
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(ps[0]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(ps[1]));
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                    _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    pp += 8;
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                    _p = _mm_mul_ps(_p, _mm_set1_ps(ps[0]));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p, _scale)))));
                    pp += 4;
                    p0++;
                    ps++;
                }

                pd += 4;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            const unsigned short* p0a = p0;
            const float* psa = ps;

            float absmax0 = 0.f;
            float absmax1 = 0.f;
            int kk = 0;
#if __SSE2__
            __m128 _absmax0 = _mm_setzero_ps();
            __m128 _absmax1 = _mm_setzero_ps();
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _s = _mm_loadu_ps(psa);
                __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                _absmax0 = _mm_max_ps(_absmax0, _mm_mul_ps(abs_ps(_p0), _s));
                __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + A_hstep)));
                _absmax1 = _mm_max_ps(_absmax1, _mm_mul_ps(abs_ps(_p1), _s));
                p0a += 4;
                psa += 4;
            }

            absmax0 = _mm_reduce_max_ps(_absmax0);
            absmax1 = _mm_reduce_max_ps(_absmax1);
#endif // __SSE2__
            for (; kk < max_kk0; kk++)
            {
                const float s = psa[0];
                absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])) * s);
                absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[A_hstep])) * s);
                p0a++;
                psa++;
            }

            float scale0 = 0.f;
            float scale1 = 0.f;
            if (absmax0 != 0.f)
                scale0 = 127.f / absmax0;
            if (absmax1 != 0.f)
                scale1 = 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            kk = 0;
#if __SSE2__
            __m128 _scale = _mm_setr_ps(scale0, scale1, 0.f, 0.f);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            __m128i _w_shift = _mm_setzero_si128();
#endif
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                __m128 _s = _mm_loadu_ps(ps);
                _p0 = _mm_mul_ps(_p0, _s);
                _p1 = _mm_mul_ps(_p1, _s);
                __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(0, 0, 0, 0)))));
                __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(1, 1, 1, 1)))));
#if __AVX512VNNI__ || __AVXVNNI__
                __m128i _q = _mm_unpacklo_epi32(_q0, _q1);
                _mm_storel_epi64((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi16(_q0, _q1));
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 8;
                p0 += 4;
                ps += 4;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk0 >= 4)
            {
                _mm_storel_epi64((__m128i*)pp, _w_shift);
                pp += 8;
            }
#endif
            for (; kk + 1 < max_kk0; kk += 2)
            {
                __m128 _p0 = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), 0.f, 0.f);
                __m128 _p1 = _mm_setr_ps(bfloat16_to_float32(p0[1]), bfloat16_to_float32(p0[A_hstep + 1]), 0.f, 0.f);
                _p0 = _mm_mul_ps(_p0, _mm_set1_ps(ps[0]));
                _p1 = _mm_mul_ps(_p1, _mm_set1_ps(ps[1]));
                __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_unpacklo_epi8(_q0, _q1)));
                pp += 4;
                p0 += 2;
                ps += 2;
            }
            for (; kk < max_kk0; kk++)
            {
                __m128 _p = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), 0.f, 0.f);
                _p = _mm_mul_ps(_p, _mm_set1_ps(ps[0]));
                unsigned int q = (unsigned int)float2int8_sse(_mm_mul_ps(_p, _scale));
                pp[0] = (signed char)q;
                pp[1] = (signed char)(q >> 8);
                pp += 2;
                p0++;
                ps++;
            }
#endif // __SSE2__
            for (; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(p0[0]);
                float v1 = bfloat16_to_float32(p0[A_hstep]);
                const float s = ps[0];
                v0 *= s;
                v1 *= s;
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
                p0++;
                ps++;
            }

            pd += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

            const unsigned short* p0a = p0;
            const float* psa = ps;
            float absmax = 0.f;
            int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _absmax512 = _mm512_setzero_ps();
            for (; kk + 15 < max_kk0; kk += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a)));
                _absmax512 = _mm512_max_ps(_absmax512, _mm512_mul_ps(abs512_ps(_p), _mm512_loadu_ps(psa)));
                p0a += 16;
                psa += 16;
            }
            absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax512));
#endif // __AVX512F__
            __m256 _absmax256 = _mm256_setzero_ps();
            for (; kk + 7 < max_kk0; kk += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a)));
                _absmax256 = _mm256_max_ps(_absmax256, _mm256_mul_ps(abs256_ps(_p), _mm256_loadu_ps(psa)));
                p0a += 8;
                psa += 8;
            }
            absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax256));
#endif // __AVX__
            __m128 _absmax128 = _mm_setzero_ps();
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                _absmax128 = _mm_max_ps(_absmax128, _mm_mul_ps(abs_ps(_p), _mm_loadu_ps(psa)));
                p0a += 4;
                psa += 4;
            }
            absmax = std::max(absmax, _mm_reduce_max_ps(_absmax128));
#endif // __SSE2__
            for (; kk < max_kk0; kk++)
            {
                absmax = std::max(absmax, fabsf(bfloat16_to_float32(p0a[0])) * psa[0]);
                p0a++;
                psa++;
            }

            if (absmax == 0.f)
            {
                pd[0] = 0.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                memset(pp, 0, max_kk0 >= 4 ? max_kk0 + 4 : max_kk0);
                pp += max_kk0 + (max_kk0 >= 4 ? 4 : 0);
#else
                memset(pp, 0, max_kk0);
                pp += max_kk0;
#endif
                p0 += max_kk0;
                ps += max_kk0;
                pd++;
                continue;
            }

            const float scale = 127.f / absmax;
            pd[0] = absmax / 127.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            int w_shift = 0;
#endif
            kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _scale512 = _mm512_set1_ps(scale);
            for (; kk + 15 < max_kk0; kk += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                _p = _mm512_mul_ps(_p, _mm512_loadu_ps(ps));
                __m128i _q = float2int8_avx512(_mm512_mul_ps(_p, _scale512));
                _mm_storeu_si128((__m128i*)pp, _q);
                pp += 16;
                p0 += 16;
                ps += 16;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _q16 = _mm256_cvtepi8_epi16(_q);
                __m256i _q32 = _mm256_madd_epi16(_q16, _mm256_set1_epi16(1));
                w_shift += _mm_reduce_add_epi32(_mm256_castsi256_si128(_q32));
                w_shift += _mm_reduce_add_epi32(_mm256_extracti128_si256(_q32, 1));
#endif
            }
#endif // __AVX512F__
            __m256 _scale256 = _mm256_set1_ps(scale);
            for (; kk + 7 < max_kk0; kk += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                _p = _mm256_mul_ps(_p, _mm256_loadu_ps(ps));
                const int64_t q = float2int8_avx(_mm256_mul_ps(_p, _scale256));
                _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(q));
                pp += 8;
                p0 += 8;
                ps += 8;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
#if defined(__x86_64__) || defined(_M_X64)
                __m128i _q8 = _mm_cvtsi64_si128(q);
#else
                __m128i _q8 = _mm_loadl_epi64((const __m128i*)(pp - 8));
#endif
                __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
            }
#endif // __AVX__
            __m128 _scale128 = _mm_set1_ps(scale);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                _p = _mm_mul_ps(_p, _mm_loadu_ps(ps));
                const int32_t q = float2int8_sse(_mm_mul_ps(_p, _scale128));
                _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(q)));
                pp += 4;
                p0 += 4;
                ps += 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _q8 = _mm_cvtsi32_si128(q);
                __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
            }
#endif // __SSE2__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk0 >= 4)
            {
                _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(w_shift * 127)));
                pp += 4;
            }
#endif
            for (; kk < max_kk0; kk++)
            {
                float v = bfloat16_to_float32(p0[0]);
                v *= ps[0];
                *pp++ = float2int8(v * scale);
                p0++;
                ps++;
            }

            pd++;
        }
    }
}

static void transpose_quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    const int elempack = A.elempack;

    if (input_scales.empty())
    {
        signed char* pp = AT_tile;
        float* pd = AT_descales_tile;
        const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
        const int block_count = (max_kk + block_size - 1) / block_size;

        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                if (elempack == 16)
                {
                    float absmax[16];
                    for (int m = 0; m < 16; m++)
                    {
                        const unsigned short* p0a = p0 + m * 16;
                        __m512 _absmax = _mm512_setzero_ps();
                        for (int kk = 0; kk < max_kk0; kk += 16)
                        {
                            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                            _absmax = _mm512_max_ps(_absmax, abs512_ps(_p));
                            p0a += A_hstep * 16;
                        }
                        absmax[m] = _mm512_comp_reduce_max_ps(_absmax);
                        pd[m] = absmax[m] / 127.f;
                    }

#if __AVX512VNNI__
                    __m512i _w_shift = _mm512_setzero_si512();
                    __m512i _v127 = _mm512_set1_epi8(127);
#endif
                    for (int kk = 0; kk < max_kk0; kk += 16)
                    {
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _mm512_set1_ps(absmax[0] == 0.f ? 0.f : 127.f / absmax[0])));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _mm512_set1_ps(absmax[1] == 0.f ? 0.f : 127.f / absmax[1])));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 32))), _mm512_set1_ps(absmax[2] == 0.f ? 0.f : 127.f / absmax[2])));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 48))), _mm512_set1_ps(absmax[3] == 0.f ? 0.f : 127.f / absmax[3])));
                        __m128i _q4 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 64))), _mm512_set1_ps(absmax[4] == 0.f ? 0.f : 127.f / absmax[4])));
                        __m128i _q5 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 80))), _mm512_set1_ps(absmax[5] == 0.f ? 0.f : 127.f / absmax[5])));
                        __m128i _q6 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 96))), _mm512_set1_ps(absmax[6] == 0.f ? 0.f : 127.f / absmax[6])));
                        __m128i _q7 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 112))), _mm512_set1_ps(absmax[7] == 0.f ? 0.f : 127.f / absmax[7])));
                        __m128i _q8 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 128))), _mm512_set1_ps(absmax[8] == 0.f ? 0.f : 127.f / absmax[8])));
                        __m128i _q9 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 144))), _mm512_set1_ps(absmax[9] == 0.f ? 0.f : 127.f / absmax[9])));
                        __m128i _qa = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 160))), _mm512_set1_ps(absmax[10] == 0.f ? 0.f : 127.f / absmax[10])));
                        __m128i _qb = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 176))), _mm512_set1_ps(absmax[11] == 0.f ? 0.f : 127.f / absmax[11])));
                        __m128i _qc = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 192))), _mm512_set1_ps(absmax[12] == 0.f ? 0.f : 127.f / absmax[12])));
                        __m128i _qd = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 208))), _mm512_set1_ps(absmax[13] == 0.f ? 0.f : 127.f / absmax[13])));
                        __m128i _qe = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 224))), _mm512_set1_ps(absmax[14] == 0.f ? 0.f : 127.f / absmax[14])));
                        __m128i _qf = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 240))), _mm512_set1_ps(absmax[15] == 0.f ? 0.f : 127.f / absmax[15])));

                        __m512i _t0 = combine4x4_epi32(_q0, _q4, _q8, _qc);
                        __m512i _t1 = combine4x4_epi32(_q1, _q5, _q9, _qd);
                        __m512i _t2 = combine4x4_epi32(_q2, _q6, _qa, _qe);
                        __m512i _t3 = combine4x4_epi32(_q3, _q7, _qb, _qf);
#if __AVX512VNNI__
                        __m512i _t4 = _mm512_unpacklo_epi32(_t0, _t1);
                        __m512i _t5 = _mm512_unpackhi_epi32(_t0, _t1);
                        __m512i _t6 = _mm512_unpacklo_epi32(_t2, _t3);
                        __m512i _t7 = _mm512_unpackhi_epi32(_t2, _t3);
                        _t0 = _mm512_unpacklo_epi64(_t4, _t6);
                        _t1 = _mm512_unpackhi_epi64(_t4, _t6);
                        _t2 = _mm512_unpacklo_epi64(_t5, _t7);
                        _t3 = _mm512_unpackhi_epi64(_t5, _t7);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _t0);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _t1);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _t2);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _t3);
#else
                        __m512i _t4 = _mm512_unpacklo_epi16(_t0, _t1);
                        __m512i _t5 = _mm512_unpackhi_epi16(_t0, _t1);
                        __m512i _t6 = _mm512_unpacklo_epi16(_t2, _t3);
                        __m512i _t7 = _mm512_unpackhi_epi16(_t2, _t3);
                        _t0 = _mm512_unpacklo_epi32(_t4, _t6);
                        _t1 = _mm512_unpackhi_epi32(_t4, _t6);
                        _t2 = _mm512_unpacklo_epi32(_t5, _t7);
                        _t3 = _mm512_unpackhi_epi32(_t5, _t7);
                        _t0 = _mm512_permutex_epi64(_t0, _MM_SHUFFLE(3, 1, 2, 0));
                        _t1 = _mm512_permutex_epi64(_t1, _MM_SHUFFLE(3, 1, 2, 0));
                        _t2 = _mm512_permutex_epi64(_t2, _MM_SHUFFLE(3, 1, 2, 0));
                        _t3 = _mm512_permutex_epi64(_t3, _MM_SHUFFLE(3, 1, 2, 0));
                        _t0 = _mm512_shuffle_i32x4(_t0, _t0, _MM_SHUFFLE(3, 1, 2, 0));
                        _t1 = _mm512_shuffle_i32x4(_t1, _t1, _MM_SHUFFLE(3, 1, 2, 0));
                        _t2 = _mm512_shuffle_i32x4(_t2, _t2, _MM_SHUFFLE(3, 1, 2, 0));
                        _t3 = _mm512_shuffle_i32x4(_t3, _t3, _MM_SHUFFLE(3, 1, 2, 0));
#endif
                        _mm512_storeu_si512((__m512i*)pp, _t0);
                        _mm512_storeu_si512((__m512i*)(pp + 64), _t1);
                        _mm512_storeu_si512((__m512i*)(pp + 128), _t2);
                        _mm512_storeu_si512((__m512i*)(pp + 192), _t3);
                        pp += 256;
                        p0 += A_hstep * 16;
                    }
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _mm512_storeu_si512((__m512i*)pp, _w_shift);
                        pp += 64;
                    }
#endif
                    pd += 16;
                }
                if (elempack == 8)
                {
                    const unsigned short* p0a = p0;
                    __m512 _absmax = _mm512_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m512 _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7;

                        _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 0 * 16)));
                        _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 1 * 16)));
                        _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 2 * 16)));
                        _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 3 * 16)));
                        _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 4 * 16)));
                        _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 5 * 16)));
                        _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 6 * 16)));
                        _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 7 * 16)));
                        transpose16x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p0));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p1));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p2));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p3));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p4));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p5));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p6));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p7));
                        p0a += A_hstep * 8;
                    }
                    __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                    __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                    __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                    _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                    __m512i _w_shift = _mm512_setzero_si512();
                    __m512i _v127 = _mm512_set1_epi8(127);
#endif
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m512 _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7;

                        _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 0 * 16)));
                        _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 1 * 16)));
                        _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 2 * 16)));
                        _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 3 * 16)));
                        _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 4 * 16)));
                        _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 5 * 16)));
                        _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 6 * 16)));
                        _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 7 * 16)));
                        transpose16x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                        _p0 = _mm512_mul_ps(_p0, _scale);
                        _p1 = _mm512_mul_ps(_p1, _scale);
                        _p2 = _mm512_mul_ps(_p2, _scale);
                        _p3 = _mm512_mul_ps(_p3, _scale);
                        _p4 = _mm512_mul_ps(_p4, _scale);
                        _p5 = _mm512_mul_ps(_p5, _scale);
                        _p6 = _mm512_mul_ps(_p6, _scale);
                        _p7 = _mm512_mul_ps(_p7, _scale);
#if __AVX512VNNI__
    {
                            __m128i _q0 = float2int8_avx512(_p0);
                            __m128i _q1 = float2int8_avx512(_p1);
                            __m128i _q2 = float2int8_avx512(_p2);
                            __m128i _q3 = float2int8_avx512(_p3);
                            transpose16x4_epi8(_q0, _q1, _q2, _q3);
                            __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                            _mm512_storeu_si512((__m512i*)pp, _q);
                            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                            pp += 64;
                        }
                        {
                            __m128i _q0 = float2int8_avx512(_p4);
                            __m128i _q1 = float2int8_avx512(_p5);
                            __m128i _q2 = float2int8_avx512(_p6);
                            __m128i _q3 = float2int8_avx512(_p7);
                            transpose16x4_epi8(_q0, _q1, _q2, _q3);
                            __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                            _mm512_storeu_si512((__m512i*)pp, _q);
                            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                            pp += 64;
                        }
#else

                        {
                            __m128i _q0 = float2int8_avx512(_p0);
                            __m128i _q1 = float2int8_avx512(_p1);
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            pp += 32;
                        }
                        {
                            __m128i _q0 = float2int8_avx512(_p2);
                            __m128i _q1 = float2int8_avx512(_p3);
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            pp += 32;
                        }
                        {
                            __m128i _q0 = float2int8_avx512(_p4);
                            __m128i _q1 = float2int8_avx512(_p5);
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            pp += 32;
                        }
                        {
                            __m128i _q0 = float2int8_avx512(_p6);
                            __m128i _q1 = float2int8_avx512(_p7);
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            pp += 32;
                        }
#endif
                        p0 += A_hstep * 8;
                    }
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _mm512_storeu_si512((__m512i*)pp, _w_shift);
                        pp += 64;
                    }
#endif
                    pd += 16;
                }
                if (elempack == 4)
                {
                    const unsigned short* p0a = p0;
                    __m512 _absmax = _mm512_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m512 _p0, _p1, _p2, _p3;

                        _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 0 * 16)));
                        _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 1 * 16)));
                        _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 2 * 16)));
                        _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 3 * 16)));
                        transpose16x4_ps(_p0, _p1, _p2, _p3);

                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p0));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p1));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p2));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p3));
                        p0a += A_hstep * 4;
                    }
                    __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                    __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                    __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                    _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                    __m512i _w_shift = _mm512_setzero_si512();
                    __m512i _v127 = _mm512_set1_epi8(127);
#endif
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m512 _p0, _p1, _p2, _p3;

                        _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 0 * 16)));
                        _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 1 * 16)));
                        _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 2 * 16)));
                        _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 3 * 16)));
                        transpose16x4_ps(_p0, _p1, _p2, _p3);

                        _p0 = _mm512_mul_ps(_p0, _scale);
                        _p1 = _mm512_mul_ps(_p1, _scale);
                        _p2 = _mm512_mul_ps(_p2, _scale);
                        _p3 = _mm512_mul_ps(_p3, _scale);
#if __AVX512VNNI__
    {
                            __m128i _q0 = float2int8_avx512(_p0);
                            __m128i _q1 = float2int8_avx512(_p1);
                            __m128i _q2 = float2int8_avx512(_p2);
                            __m128i _q3 = float2int8_avx512(_p3);
                            transpose16x4_epi8(_q0, _q1, _q2, _q3);
                            __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                            _mm512_storeu_si512((__m512i*)pp, _q);
                            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                            pp += 64;
                        }
#else

                        {
                            __m128i _q0 = float2int8_avx512(_p0);
                            __m128i _q1 = float2int8_avx512(_p1);
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            pp += 32;
                        }
                        {
                            __m128i _q0 = float2int8_avx512(_p2);
                            __m128i _q1 = float2int8_avx512(_p3);
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                            pp += 32;
                        }
#endif
                        p0 += A_hstep * 4;
                    }
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _mm512_storeu_si512((__m512i*)pp, _w_shift);
                        pp += 64;
                    }
#endif
                    pd += 16;
                }

                if (elempack == 1)
                {
                    const unsigned short* p0a = p0;

                    __m512 _absmax = _mm512_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a)));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p));
                        p0a += A_hstep;
                    }

                    __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                    __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                    __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                    _mm512_storeu_ps(pd, _descale);

#if __AVX512VNNI__
                    __m512i _w_shift = _mm512_setzero_si512();
                    __m512i _v127 = _mm512_set1_epi8(127);
#endif
                    int kk = 0;
#if __AVX512VNNI__
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                        __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + A_hstep)));
                        __m512 _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + A_hstep * 2)));
                        __m512 _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + A_hstep * 3)));
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_p2, _scale));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_p3, _scale));
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                        p0 += A_hstep * 4;
                    }
#endif // __AVX512VNNI__
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _mm512_storeu_si512((__m512i*)pp, _w_shift);
                        pp += 64;
                    }
#endif // __AVX512VNNI__
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                        __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + A_hstep)));
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                        _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                        pp += 16;
                        p0 += A_hstep;
                    }

                    pd += 16;
                }
            }
        }
#endif // __AVX512F__
#if !__AVX2__
        signed char* pp1 = pp + AT_tile.w * 4;
        float* pd1 = pd + AT_descales_tile.w * 4;
#endif
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __AVX512F__
                if (elempack == 16)
                {
                    float absmax[8];
                    for (int m = 0; m < 8; m++)
                    {
                        const unsigned short* p0a = p0 + m * 16;
                        __m512 _absmax = _mm512_setzero_ps();
                        for (int kk = 0; kk < max_kk0; kk += 16)
                        {
                            _absmax = _mm512_max_ps(_absmax, abs512_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a))));
                            p0a += A_hstep * 16;
                        }
                        absmax[m] = _mm512_comp_reduce_max_ps(_absmax);
                        pd[m] = absmax[m] / 127.f;
                    }
#if __AVX512VNNI__
                    __m512i _w_shift512 = _mm512_setzero_si512();
#endif
                    for (int kk = 0; kk < max_kk0; kk += 16)
                    {
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _mm512_set1_ps(absmax[0] == 0.f ? 0.f : 127.f / absmax[0])));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _mm512_set1_ps(absmax[1] == 0.f ? 0.f : 127.f / absmax[1])));
                        __m128i _q2 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 32))), _mm512_set1_ps(absmax[2] == 0.f ? 0.f : 127.f / absmax[2])));
                        __m128i _q3 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 48))), _mm512_set1_ps(absmax[3] == 0.f ? 0.f : 127.f / absmax[3])));
                        __m128i _q4 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 64))), _mm512_set1_ps(absmax[4] == 0.f ? 0.f : 127.f / absmax[4])));
                        __m128i _q5 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 80))), _mm512_set1_ps(absmax[5] == 0.f ? 0.f : 127.f / absmax[5])));
                        __m128i _q6 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 96))), _mm512_set1_ps(absmax[6] == 0.f ? 0.f : 127.f / absmax[6])));
                        __m128i _q7 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 112))), _mm512_set1_ps(absmax[7] == 0.f ? 0.f : 127.f / absmax[7])));
#if __AVX512VNNI__
                        transpose4x8_epi32(_q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7);
#else
                        transpose8x8_epi16(_q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7);
#endif
                        __m512i _t0 = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        __m512i _t1 = combine4x4_epi32(_q4, _q5, _q6, _q7);
#if __AVX512VNNI__
                        _w_shift512 = _mm512_dpbusd_epi32(_w_shift512, _mm512_set1_epi8(127), _t0);
                        _w_shift512 = _mm512_dpbusd_epi32(_w_shift512, _mm512_set1_epi8(127), _t1);
#endif
                        _mm512_storeu_si512((__m512i*)pp, _t0);
                        _mm512_storeu_si512((__m512i*)(pp + 64), _t1);
                        pp += 128;
                        p0 += A_hstep * 16;
                    }
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        __m256i _w_shift = _mm256_add_epi32(_mm512_extracti32x8_epi32(_w_shift512, 0), _mm512_extracti32x8_epi32(_w_shift512, 1));
                        _mm256_storeu_si256((__m256i*)pp, _w_shift);
                        pp += 32;
                    }
#endif
                    pd += 8;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    const unsigned short* p0a = p0;
                    __m256 _absmax = _mm256_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m256 _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7;

                        _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 0 * 8)));
                        _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 1 * 8)));
                        _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 2 * 8)));
                        _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 3 * 8)));
                        _p4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 4 * 8)));
                        _p5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 5 * 8)));
                        _p6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 6 * 8)));
                        _p7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 7 * 8)));
                        transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p0));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p1));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p2));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p3));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p4));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p5));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p6));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p7));
                        p0a += A_hstep * 8;
                    }
                    __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                    __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                    __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                    __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                    _mm256_storeu_ps(pd, _descale);
#else
                    _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                    _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m256i _w_shift = _mm256_setzero_si256();
#endif
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m256 _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7;

                        _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 0 * 8)));
                        _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 1 * 8)));
                        _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 2 * 8)));
                        _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 3 * 8)));
                        _p4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 4 * 8)));
                        _p5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 5 * 8)));
                        _p6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 6 * 8)));
                        _p7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 7 * 8)));
                        transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                        _p0 = _mm256_mul_ps(_p0, _scale);
                        _p1 = _mm256_mul_ps(_p1, _scale);
                        _p2 = _mm256_mul_ps(_p2, _scale);
                        _p3 = _mm256_mul_ps(_p3, _scale);
                        _p4 = _mm256_mul_ps(_p4, _scale);
                        _p5 = _mm256_mul_ps(_p5, _scale);
                        _p6 = _mm256_mul_ps(_p6, _scale);
                        _p7 = _mm256_mul_ps(_p7, _scale);
#if __AVX512VNNI__ || __AVXVNNI__
    {
                            __m128i _q0 = _mm_cvtsi64_si128(float2int8_avx(_p0));
                            __m128i _q1 = _mm_cvtsi64_si128(float2int8_avx(_p1));
                            __m128i _q2 = _mm_cvtsi64_si128(float2int8_avx(_p2));
                            __m128i _q3 = _mm_cvtsi64_si128(float2int8_avx(_p3));
                            transpose8x4_epi8(_q0, _q1, _q2, _q3);
                            __m256i _q = combine4x2_epi32(_q0, _q1);
                            _mm256_storeu_si256((__m256i*)pp, _q);
#if !__AVXVNNIINT8__
                            _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
                            pp += 32;
                        }
                        {
                            __m128i _q0 = _mm_cvtsi64_si128(float2int8_avx(_p4));
                            __m128i _q1 = _mm_cvtsi64_si128(float2int8_avx(_p5));
                            __m128i _q2 = _mm_cvtsi64_si128(float2int8_avx(_p6));
                            __m128i _q3 = _mm_cvtsi64_si128(float2int8_avx(_p7));
                            transpose8x4_epi8(_q0, _q1, _q2, _q3);
                            __m256i _q = combine4x2_epi32(_q0, _q1);
                            _mm256_storeu_si256((__m256i*)pp, _q);
#if !__AVXVNNIINT8__
                            _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
                            pp += 32;
                        }
#else

#if __AVX2__
                        {
                            __m128i _q0 = _mm_cvtsi64_si128(float2int8_avx(_p0));
                            __m128i _q1 = _mm_cvtsi64_si128(float2int8_avx(_p1));
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            pp += 16;
                        }
                        {
                            __m128i _q0 = _mm_cvtsi64_si128(float2int8_avx(_p2));
                            __m128i _q1 = _mm_cvtsi64_si128(float2int8_avx(_p3));
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            pp += 16;
                        }
                        {
                            __m128i _q0 = _mm_cvtsi64_si128(float2int8_avx(_p4));
                            __m128i _q1 = _mm_cvtsi64_si128(float2int8_avx(_p5));
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            pp += 16;
                        }
                        {
                            __m128i _q0 = _mm_cvtsi64_si128(float2int8_avx(_p6));
                            __m128i _q1 = _mm_cvtsi64_si128(float2int8_avx(_p7));
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                            pp += 16;
                        }
#else
                        __m128i _q0 = _mm_cvtsi64_si128(float2int8_avx(_p0));
                        __m128i _q1 = _mm_cvtsi64_si128(float2int8_avx(_p1));
                        __m128i _q2 = _mm_cvtsi64_si128(float2int8_avx(_p2));
                        __m128i _q3 = _mm_cvtsi64_si128(float2int8_avx(_p3));
                        __m128i _q4 = _mm_cvtsi64_si128(float2int8_avx(_p4));
                        __m128i _q5 = _mm_cvtsi64_si128(float2int8_avx(_p5));
                        __m128i _q6 = _mm_cvtsi64_si128(float2int8_avx(_p6));
                        __m128i _q7 = _mm_cvtsi64_si128(float2int8_avx(_p7));
                        __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                        __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
                        __m128i _q45 = _mm_unpacklo_epi8(_q4, _q5);
                        __m128i _q67 = _mm_unpacklo_epi8(_q6, _q7);
                        _mm_storel_epi64((__m128i*)pp, _q01);
                        _mm_storel_epi64((__m128i*)(pp + 8), _q23);
                        _mm_storel_epi64((__m128i*)(pp + 16), _q45);
                        _mm_storel_epi64((__m128i*)(pp + 24), _q67);
                        _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q01));
                        _mm_storeh_pd((double*)(pp1 + 8), _mm_castsi128_pd(_q23));
                        _mm_storeh_pd((double*)(pp1 + 16), _mm_castsi128_pd(_q45));
                        _mm_storeh_pd((double*)(pp1 + 24), _mm_castsi128_pd(_q67));
                        pp += 32;
                        pp1 += 32;
#endif // __AVX2__
#endif
                        p0 += A_hstep * 8;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm256_storeu_si256((__m256i*)pp, _w_shift);
                        pp += 32;
                    }
#endif
#if __AVX2__
                    pd += 8;
#else
                    pd += 4;
                    pd1 += 4;
#endif
                }
                if (elempack == 4)
                {
                    float absmax[8];
                    for (int m = 0; m < 8; m++)
                    {
                        const unsigned short* p0a = p0 + m * 4;
                        __m128 _absmax = _mm_setzero_ps();
                        for (int kk = 0; kk < max_kk0; kk += 4)
                        {
                            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p));
                            p0a += A_hstep * 4;
                        }
                        absmax[m] = _mm_reduce_max_ps(_absmax);
                    }
                    __m256 _descale = _mm256_div_ps(_mm256_loadu_ps(absmax), _mm256_set1_ps(127.f));
#if __AVX2__
                    _mm256_storeu_ps(pd, _descale);
#else
                    _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                    _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif

                    __m256 _scale0 = combine4x2_ps(_mm_set1_ps(absmax[0] == 0.f ? 0.f : 127.f / absmax[0]), _mm_set1_ps(absmax[1] == 0.f ? 0.f : 127.f / absmax[1]));
                    __m256 _scale1 = combine4x2_ps(_mm_set1_ps(absmax[2] == 0.f ? 0.f : 127.f / absmax[2]), _mm_set1_ps(absmax[3] == 0.f ? 0.f : 127.f / absmax[3]));
                    __m256 _scale2 = combine4x2_ps(_mm_set1_ps(absmax[4] == 0.f ? 0.f : 127.f / absmax[4]), _mm_set1_ps(absmax[5] == 0.f ? 0.f : 127.f / absmax[5]));
                    __m256 _scale3 = combine4x2_ps(_mm_set1_ps(absmax[6] == 0.f ? 0.f : 127.f / absmax[6]), _mm_set1_ps(absmax[7] == 0.f ? 0.f : 127.f / absmax[7]));
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m256i _w_shift = _mm256_setzero_si256();
#endif
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m256 _p0 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _scale0);
                        __m256 _p1 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _scale1);
                        __m256 _p2 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 16))), _scale2);
                        __m256 _p3 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 24))), _scale3);
                        __m128i _q0 = float2int8_avx(_p0, _p1);
                        __m128i _q1 = float2int8_avx(_p2, _p3);
#if __AVX512VNNI__ || __AVXVNNI__
                        __m256i _q = combine4x2_epi32(_q0, _q1);
                        _mm256_storeu_si256((__m256i*)pp, _q);
#if !__AVXVNNIINT8__
                        _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
                        pp += 32;
#else
#if __AVX2__
                        __m128i _t0 = _mm_unpacklo_epi16(_q0, _q1);
                        __m128i _t1 = _mm_unpackhi_epi16(_q0, _q1);
                        __m128i _t2 = _mm_unpacklo_epi16(_t0, _t1);
                        __m128i _t3 = _mm_unpackhi_epi16(_t0, _t1);
                        _t0 = _mm_unpacklo_epi16(_t2, _t3);
                        _t1 = _mm_unpackhi_epi16(_t2, _t3);
                        _mm_storeu_si128((__m128i*)pp, _t0);
                        _mm_storeu_si128((__m128i*)(pp + 16), _t1);
                        pp += 32;
#else
                        __m128i _si = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                        _mm_storeu_si128((__m128i*)pp, _mm_shuffle_epi8(_q0, _si));
                        _mm_storeu_si128((__m128i*)pp1, _mm_shuffle_epi8(_q1, _si));
                        pp += 16;
                        pp1 += 16;
#endif
#endif
                        p0 += A_hstep * 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm256_storeu_si256((__m256i*)pp, _w_shift);
                        pp += 32;
                    }
#endif
#if __AVX2__
                    pd += 8;
#else
                    pd += 4;
                    pd1 += 4;
#endif
                }
                if (elempack == 1)
                {
                    const unsigned short* p0a = p0;

                    __m256 _absmax = _mm256_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a)));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p));
                        p0a += A_hstep;
                    }

                    __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                    __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                    __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                    __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                    _mm256_storeu_ps(pd, _descale);
#else
                    _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                    _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m256i _w_shift = _mm256_setzero_si256();
#endif
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep)));
                        __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2)));
                        __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3)));
                        _p0 = _mm256_mul_ps(_p0, _scale);
                        _p1 = _mm256_mul_ps(_p1, _scale);
                        _p2 = _mm256_mul_ps(_p2, _scale);
                        _p3 = _mm256_mul_ps(_p3, _scale);

                        __m128i _q0 = float2int8_avx(_p0, _p2);
                        __m128i _q1 = float2int8_avx(_p1, _p3);
                        __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                        __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                        _q0 = _mm_unpacklo_epi16(_q01, _q23);
                        _q1 = _mm_unpackhi_epi16(_q01, _q23);
                        __m256i _q = combine4x2_epi32(_q0, _q1);
                        _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                        _mm_storeu_si128((__m128i*)pp, _q01);
                        _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                        _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX2__
                        pp += 32;
#else
                        pp += 16;
                        pp1 += 16;
#endif
                        p0 += A_hstep * 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm256_storeu_si256((__m256i*)pp, _w_shift);
                        pp += 32;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep)));
                        _p0 = _mm256_mul_ps(_p0, _scale);
                        _p1 = _mm256_mul_ps(_p1, _scale);
                        __m128i _q = float2int8_avx(_p0, _p1);
                        __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                        _q = _mm_shuffle_epi8(_q, _si);
#if __AVX2__
                        _mm_storeu_si128((__m128i*)pp, _q);
                        pp += 16;
#else
                        _mm_storel_pd((double*)pp, _mm_castsi128_pd(_q));
                        _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q));
                        pp += 8;
                        pp1 += 8;
#endif
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
#if __AVX2__
                        _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(float2int8_avx(_mm256_mul_ps(_p, _scale))));
                        pp += 8;
#else
                        const uint64_t q = (uint64_t)float2int8_avx(_mm256_mul_ps(_p, _scale));
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128((int)q)));
                        _mm_store_ss((float*)pp1, _mm_castsi128_ps(_mm_cvtsi32_si128((int)(q >> 32))));
                        pp += 4;
                        pp1 += 4;
#endif
                        p0 += A_hstep;
                    }

#if __AVX2__
                    pd += 8;
#else
                    pd += 4;
                    pd1 += 4;
#endif
                }
            }
#if !__AVX2__
            pp = pp1;
            pp1 = pp + AT_tile.w * 4;
            pd = pd1;
            pd1 = pd + AT_descales_tile.w * 4;
#endif
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    const unsigned short* p0a = p0;
                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 16)
                    {
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 16)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 32)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 48)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p1));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p2));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p3));
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 4)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 20)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 36)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 52)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p1));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p2));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p3));
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 8)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 24)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 40)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 56)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p1));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p2));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p3));
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 12)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 28)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 44)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 60)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p1));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p2));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p3));
                        }
                        p0a += A_hstep * 16;
                    }
                    __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                    __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                    __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                    __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                    _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
#endif
                    for (int kk = 0; kk < max_kk0; kk += 16)
                    {
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 16)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 32)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 48)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _p0 = _mm_mul_ps(_p0, _scale);
                            _p1 = _mm_mul_ps(_p1, _scale);
                            _p2 = _mm_mul_ps(_p2, _scale);
                            _p3 = _mm_mul_ps(_p3, _scale);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
                            __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_p2));
                            __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_p3));
                            __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                            __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                            _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif
                            pp += 16;
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 20)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 36)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 52)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _p0 = _mm_mul_ps(_p0, _scale);
                            _p1 = _mm_mul_ps(_p1, _scale);
                            _p2 = _mm_mul_ps(_p2, _scale);
                            _p3 = _mm_mul_ps(_p3, _scale);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
                            __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_p2));
                            __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_p3));
                            __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                            __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                            _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif
                            pp += 16;
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 8)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 24)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 40)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 56)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _p0 = _mm_mul_ps(_p0, _scale);
                            _p1 = _mm_mul_ps(_p1, _scale);
                            _p2 = _mm_mul_ps(_p2, _scale);
                            _p3 = _mm_mul_ps(_p3, _scale);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
                            __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_p2));
                            __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_p3));
                            __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                            __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                            _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif
                            pp += 16;
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 12)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 28)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 44)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 60)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _p0 = _mm_mul_ps(_p0, _scale);
                            _p1 = _mm_mul_ps(_p1, _scale);
                            _p2 = _mm_mul_ps(_p2, _scale);
                            _p3 = _mm_mul_ps(_p3, _scale);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
                            __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_p2));
                            __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_p3));
                            __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                            __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                            _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif
                            pp += 16;
                        }
                        p0 += A_hstep * 16;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_storeu_si128((__m128i*)pp, _w_shift);
                        pp += 16;
                    }
#endif
                    pd += 4;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    const unsigned short* p0a = p0;
                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 8)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 16)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 24)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p1));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p2));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p3));
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 4)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 12)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 20)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 28)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p1));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p2));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p3));
                        }
                        p0a += A_hstep * 8;
                    }
                    __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                    __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                    __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                    __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                    _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
#endif
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 8)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 16)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 24)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _p0 = _mm_mul_ps(_p0, _scale);
                            _p1 = _mm_mul_ps(_p1, _scale);
                            _p2 = _mm_mul_ps(_p2, _scale);
                            _p3 = _mm_mul_ps(_p3, _scale);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
                            __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_p2));
                            __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_p3));
                            __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                            __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                            _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif
                            pp += 16;
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 12)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 20)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 28)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _p0 = _mm_mul_ps(_p0, _scale);
                            _p1 = _mm_mul_ps(_p1, _scale);
                            _p2 = _mm_mul_ps(_p2, _scale);
                            _p3 = _mm_mul_ps(_p3, _scale);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
                            __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_p2));
                            __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_p3));
                            __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                            __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                            _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif
                            pp += 16;
                        }
                        p0 += A_hstep * 8;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_storeu_si128((__m128i*)pp, _w_shift);
                        pp += 16;
                    }
#endif
                    pd += 4;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    const unsigned short* p0a = p0;
                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 4)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 8)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 12)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p1));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p2));
                            _absmax = _mm_max_ps(_absmax, abs_ps(_p3));
                        }
                        p0a += A_hstep * 4;
                    }
                    __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                    __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                    __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                    __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                    _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
#endif
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4)));
                            __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 8)));
                            __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 12)));
                            __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                            __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                            __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                            __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);
                            _p0 = _mm_movelh_ps(_t0, _t2);
                            _p1 = _mm_movehl_ps(_t2, _t0);
                            _p2 = _mm_movelh_ps(_t1, _t3);
                            _p3 = _mm_movehl_ps(_t3, _t1);
                            _p0 = _mm_mul_ps(_p0, _scale);
                            _p1 = _mm_mul_ps(_p1, _scale);
                            _p2 = _mm_mul_ps(_p2, _scale);
                            _p3 = _mm_mul_ps(_p3, _scale);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
                            __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_p2));
                            __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_p3));
                            __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                            __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                            _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                            _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif
                            pp += 16;
                        }
                        p0 += A_hstep * 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_storeu_si128((__m128i*)pp, _w_shift);
                        pp += 16;
                    }
#endif
                    pd += 4;
                }

                if (elempack == 1)
                {
                    const unsigned short* p0a = p0;

                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                        _absmax = _mm_max_ps(_absmax, abs_ps(_p));
                        p0a += A_hstep;
                    }

                    __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                    __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                    __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                    __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                    _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
#endif
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                        __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                        __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                        __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                        __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                        __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p2, _scale)));
                        __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p3, _scale)));
                        __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                        __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                        __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                        _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_storeu_si128((__m128i*)pp, _w_shift);
                        pp += 16;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                        __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                        __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                        _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        pp += 8;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p, _scale)))));
                        pp += 4;
                        p0 += A_hstep;
                    }

                    pd += 4;
                }
            }
        }
#endif // __SSE2__
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    const unsigned short* p0a = p0;
                    __m512 _absmax0 = _mm512_setzero_ps();
                    __m512 _absmax1 = _mm512_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 16)
                    {
                        __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                        __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 16)));
                        _absmax0 = _mm512_max_ps(_absmax0, abs512_ps(_p0));
                        _absmax1 = _mm512_max_ps(_absmax1, abs512_ps(_p1));
                        p0a += A_hstep * 16;
                    }

                    float absmax0 = _mm512_comp_reduce_max_ps(_absmax0);
                    float absmax1 = _mm512_comp_reduce_max_ps(_absmax1);
                    float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                    float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    __m512 _scale0 = _mm512_set1_ps(scale0);
                    __m512 _scale1 = _mm512_set1_ps(scale1);
#if __AVX512VNNI__
                    __m128i _w_shift = _mm_setzero_si128();
                    __m128i _v127 = _mm_set1_epi8(127);
#endif
                    for (int kk = 0; kk < max_kk0; kk += 16)
                    {
                        __m128i _q0 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _scale0));
                        __m128i _q1 = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _scale1));
#if __AVX512VNNI__
                        __m128i _t0 = _mm_unpacklo_epi32(_q0, _q1);
                        __m128i _t1 = _mm_unpackhi_epi32(_q0, _q1);
                        _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _t0);
                        _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _t1);
#else
                        __m128i _t0 = _mm_unpacklo_epi16(_q0, _q1);
                        __m128i _t1 = _mm_unpackhi_epi16(_q0, _q1);
#endif
                        _mm_storeu_si128((__m128i*)pp, _t0);
                        _mm_storeu_si128((__m128i*)(pp + 16), _t1);
                        pp += 32;
                        p0 += A_hstep * 16;
                    }
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        _w_shift = _mm_shuffle_epi32(_w_shift, _MM_SHUFFLE(3, 1, 2, 0));
                        _w_shift = _mm_hadd_epi32(_w_shift, _w_shift);
                        _mm_storel_epi64((__m128i*)pp, _w_shift);
                        pp += 8;
                    }
#endif

                    pd += 2;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    const unsigned short* p0a = p0;
                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 8)));
                            __m128 _a0 = _mm_set_ss(_mm_reduce_max_ps(abs_ps(_p0)));
                            __m128 _a1 = _mm_set_ss(_mm_reduce_max_ps(abs_ps(_p1)));
                            _absmax = _mm_max_ps(_absmax, _mm_unpacklo_ps(_a0, _a1));
                        }
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 4)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 12)));
                            __m128 _a0 = _mm_set_ss(_mm_reduce_max_ps(abs_ps(_p0)));
                            __m128 _a1 = _mm_set_ss(_mm_reduce_max_ps(abs_ps(_p1)));
                            _absmax = _mm_max_ps(_absmax, _mm_unpacklo_ps(_a0, _a1));
                        }
                        p0a += A_hstep * 8;
                    }
                    float absmax0 = _mm_cvtss_f32(_absmax);
                    float absmax1 = _mm_cvtss_f32(_mm_shuffle_ps(_absmax, _absmax, _MM_SHUFFLE(1, 1, 1, 1)));
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    __m128 _scale = _mm_setr_ps(absmax0 == 0.f ? 0.f : 127.f / absmax0, absmax1 == 0.f ? 0.f : 127.f / absmax1, 0.f, 0.f);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
#endif
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        {
                            __m128 _scale0 = _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(0, 0, 0, 0));
                            __m128 _scale1 = _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(1, 1, 1, 1));
                            __m128 _p0 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0))), _scale0);
                            __m128 _p1 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 8))), _scale1);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi32(_q0, _q1);
#else
                            __m128i _q = _mm_unpacklo_epi16(_q0, _q1);
#endif
                            _mm_storel_epi64((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
                            pp += 8;
                        }
                        {
                            __m128 _scale0 = _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(0, 0, 0, 0));
                            __m128 _scale1 = _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(1, 1, 1, 1));
                            __m128 _p0 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))), _scale0);
                            __m128 _p1 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 12))), _scale1);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi32(_q0, _q1);
#else
                            __m128i _q = _mm_unpacklo_epi16(_q0, _q1);
#endif
                            _mm_storel_epi64((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
                            pp += 8;
                        }
                        p0 += A_hstep * 8;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_storel_epi64((__m128i*)pp, _w_shift);
                        pp += 8;
                    }
#endif
                    pd += 2;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    const unsigned short* p0a = p0;
                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        {
                            __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                            __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 4)));
                            __m128 _a0 = _mm_set_ss(_mm_reduce_max_ps(abs_ps(_p0)));
                            __m128 _a1 = _mm_set_ss(_mm_reduce_max_ps(abs_ps(_p1)));
                            _absmax = _mm_max_ps(_absmax, _mm_unpacklo_ps(_a0, _a1));
                        }
                        p0a += A_hstep * 4;
                    }
                    float absmax0 = _mm_cvtss_f32(_absmax);
                    float absmax1 = _mm_cvtss_f32(_mm_shuffle_ps(_absmax, _absmax, _MM_SHUFFLE(1, 1, 1, 1)));
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    __m128 _scale = _mm_setr_ps(absmax0 == 0.f ? 0.f : 127.f / absmax0, absmax1 == 0.f ? 0.f : 127.f / absmax1, 0.f, 0.f);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
#endif
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        {
                            __m128 _scale0 = _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(0, 0, 0, 0));
                            __m128 _scale1 = _mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(1, 1, 1, 1));
                            __m128 _p0 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0))), _scale0);
                            __m128 _p1 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))), _scale1);
                            __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                            __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
#if __AVX512VNNI__ || __AVXVNNI__
                            __m128i _q = _mm_unpacklo_epi32(_q0, _q1);
#else
                            __m128i _q = _mm_unpacklo_epi16(_q0, _q1);
#endif
                            _mm_storel_epi64((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
                            pp += 8;
                        }
                        p0 += A_hstep * 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_storel_epi64((__m128i*)pp, _w_shift);
                        pp += 8;
                    }
#endif
                    pd += 2;
                }
#endif // __SSE2__
                if (elempack == 1)
                {
                    const unsigned short* p0a = p0;

                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    int kk = 0;
#if __SSE2__
                    __m128 _absmax = _mm_setzero_ps();
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)p0a)));
                        _absmax = _mm_max_ps(_absmax, abs_ps(_p));
                        p0a += A_hstep;
                    }

                    absmax0 = _mm_cvtss_f32(_absmax);
                    absmax1 = _mm_cvtss_f32(_mm_shuffle_ps(_absmax, _absmax, _MM_SHUFFLE(1, 1, 1, 1)));
#endif // __SSE2__
                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, (float)fabsf(bfloat16_to_float32(p0a[0])));
                        absmax1 = std::max(absmax1, (float)fabsf(bfloat16_to_float32(p0a[1])));
                        p0a += A_hstep;
                    }

                    float scale0 = 0.f;
                    float scale1 = 0.f;
                    if (absmax0 != 0.f)
                        scale0 = 127.f / absmax0;
                    if (absmax1 != 0.f)
                        scale1 = 127.f / absmax1;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    kk = 0;
#if __SSE2__
                    __m128 _scale = _mm_setr_ps(scale0, scale1, 0.f, 0.f);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _w_shift = _mm_setzero_si128();
#endif
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)p0)));
                        __m128 _p1 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)(p0 + A_hstep))));
                        __m128 _p2 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)(p0 + A_hstep * 2))));
                        __m128 _p3 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)(p0 + A_hstep * 3))));
                        __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                        __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                        __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p2, _scale)));
                        __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p3, _scale)));
                        __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                        __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                        __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                        _mm_storel_epi64((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                        _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi32(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
                        pp += 8;
                        p0 += A_hstep * 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_storel_epi64((__m128i*)pp, _w_shift);
                        pp += 8;
                    }
#endif
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        __m128 _p0 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)p0)));
                        __m128 _p1 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)(p0 + A_hstep))));
                        __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                        __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_unpacklo_epi8(_q0, _q1)));
                        pp += 4;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)p0)));
                        unsigned int q = (unsigned int)float2int8_sse(_mm_mul_ps(_p, _scale));
                        pp[0] = (signed char)q;
                        pp[1] = (signed char)(q >> 8);
                        pp += 2;
                        p0 += A_hstep;
                    }
#endif // __SSE2__
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[1]);
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale1);
                        pp += 2;
                        p0 += A_hstep;
                    }

                    pd += 2;
                }
            }
        }

        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    const unsigned short* p0a = p0;
                    __m512 _absmax = _mm512_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 16)
                    {
                        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                        _absmax = _mm512_max_ps(_absmax, abs512_ps(_p));
                        p0a += A_hstep * 16;
                    }

                    float absmax = _mm512_comp_reduce_max_ps(_absmax);
                    float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                    pd[0] = absmax / 127.f;
                    __m512 _scale = _mm512_set1_ps(scale);
#if __AVX512VNNI__
                    int w_shift = 0;
#endif
                    for (int kk = 0; kk < max_kk0; kk += 16)
                    {
                        __m128i _q = float2int8_avx512(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _scale));
                        _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__
                        __m256i _q16 = _mm256_cvtepi8_epi16(_q);
                        __m256i _q32 = _mm256_madd_epi16(_q16, _mm256_set1_epi16(1));
                        w_shift += _mm_reduce_add_epi32(_mm256_castsi256_si128(_q32));
                        w_shift += _mm_reduce_add_epi32(_mm256_extracti128_si256(_q32, 1));
#endif
                        pp += 16;
                        p0 += A_hstep * 16;
                    }
#if __AVX512VNNI__
                    if (max_kk0 >= 4)
                    {
                        ((int*)pp)[0] = w_shift * 127;
                        pp += 4;
                    }
#endif
                    pd++;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    const unsigned short* p0a = p0;
                    float absmax = 0.f;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        {
                            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                            absmax = std::max(absmax, _mm_reduce_max_ps(abs_ps(_p)));
                        }
                        {
                            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 4)));
                            absmax = std::max(absmax, _mm_reduce_max_ps(abs_ps(_p)));
                        }
                        p0a += A_hstep * 8;
                    }
                    float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                    pd[0] = absmax / 127.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    int w_shift = 0;
#endif
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        {
                            __m128 _p = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0))), _mm_set1_ps(scale));
                            const int32_t q = float2int8_sse(_p);
                            _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(q)));
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            __m128i _q8 = _mm_cvtsi32_si128(q);
                            __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                            w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                            pp += 4;
                        }
                        {
                            __m128 _p = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))), _mm_set1_ps(scale));
                            const int32_t q = float2int8_sse(_p);
                            _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(q)));
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            __m128i _q8 = _mm_cvtsi32_si128(q);
                            __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                            w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                            pp += 4;
                        }
                        p0 += A_hstep * 8;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        ((int*)pp)[0] = w_shift * 127;
                        pp += 4;
                    }
#endif
                    pd++;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    const unsigned short* p0a = p0;
                    float absmax = 0.f;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        {
                            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                            absmax = std::max(absmax, _mm_reduce_max_ps(abs_ps(_p)));
                        }
                        p0a += A_hstep * 4;
                    }
                    float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                    pd[0] = absmax / 127.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    int w_shift = 0;
#endif
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        {
                            __m128 _p = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0))), _mm_set1_ps(scale));
                            const int32_t q = float2int8_sse(_p);
                            _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(q)));
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                            __m128i _q8 = _mm_cvtsi32_si128(q);
                            __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                            w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                            pp += 4;
                        }
                        p0 += A_hstep * 4;
                    }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        ((int*)pp)[0] = w_shift * 127;
                        pp += 4;
                    }
#endif
                    pd++;
                }
#endif // __SSE2__
                if (elempack == 1)
                {
                    const unsigned short* p0a = p0;
                    float absmax = 0.f;
                    int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _absmax512 = _mm512_setzero_ps();
                    for (; kk + 15 < max_kk0; kk += 16)
                    {
                        __m512 _p = _mm512_setr_ps(bfloat16_to_float32(p0a[0]), bfloat16_to_float32(p0a[A_hstep]), bfloat16_to_float32(p0a[A_hstep * 2]), bfloat16_to_float32(p0a[A_hstep * 3]), bfloat16_to_float32(p0a[A_hstep * 4]), bfloat16_to_float32(p0a[A_hstep * 5]), bfloat16_to_float32(p0a[A_hstep * 6]), bfloat16_to_float32(p0a[A_hstep * 7]), bfloat16_to_float32(p0a[A_hstep * 8]), bfloat16_to_float32(p0a[A_hstep * 9]), bfloat16_to_float32(p0a[A_hstep * 10]), bfloat16_to_float32(p0a[A_hstep * 11]), bfloat16_to_float32(p0a[A_hstep * 12]), bfloat16_to_float32(p0a[A_hstep * 13]), bfloat16_to_float32(p0a[A_hstep * 14]), bfloat16_to_float32(p0a[A_hstep * 15]));
                        _absmax512 = _mm512_max_ps(_absmax512, abs512_ps(_p));
                        p0a += A_hstep * 16;
                    }
                    absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax512));
#endif // __AVX512F__
#if __AVX2__
                    __m256 _absmax256 = _mm256_setzero_ps();
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _p = _mm256_setr_ps(bfloat16_to_float32(p0a[0]), bfloat16_to_float32(p0a[A_hstep]), bfloat16_to_float32(p0a[A_hstep * 2]), bfloat16_to_float32(p0a[A_hstep * 3]), bfloat16_to_float32(p0a[A_hstep * 4]), bfloat16_to_float32(p0a[A_hstep * 5]), bfloat16_to_float32(p0a[A_hstep * 6]), bfloat16_to_float32(p0a[A_hstep * 7]));
                        _absmax256 = _mm256_max_ps(_absmax256, abs256_ps(_p));
                        p0a += A_hstep * 8;
                    }
                    absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax256));
#endif // __AVX2__
#endif // __AVX__
                    __m128 _absmax128 = _mm_setzero_ps();
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p = _mm_setr_ps(bfloat16_to_float32(p0a[0]), bfloat16_to_float32(p0a[A_hstep]), bfloat16_to_float32(p0a[A_hstep * 2]), bfloat16_to_float32(p0a[A_hstep * 3]));
                        _absmax128 = _mm_max_ps(_absmax128, abs_ps(_p));
                        p0a += A_hstep * 4;
                    }
                    absmax = std::max(absmax, _mm_reduce_max_ps(_absmax128));
#endif // __SSE2__
                    for (; kk < max_kk0; kk++)
                    {
                        float v = bfloat16_to_float32(p0a[0]);
                        absmax = std::max(absmax, (float)fabsf(v));
                        p0a += A_hstep;
                    }

                    if (absmax == 0.f)
                    {
                        pd[0] = 0.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        memset(pp, 0, max_kk0 >= 4 ? max_kk0 + 4 : max_kk0);
                        pp += max_kk0 + (max_kk0 >= 4 ? 4 : 0);
#else
                        memset(pp, 0, max_kk0);
                        pp += max_kk0;
#endif
                        p0 += max_kk0 * A_hstep;
                        pd++;
                        continue;
                    }

                    const float scale = 127.f / absmax;
                    pd[0] = absmax / 127.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    int w_shift = 0;
#endif
                    kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _scale512 = _mm512_set1_ps(scale);
                    for (; kk + 15 < max_kk0; kk += 16)
                    {
                        __m512 _p = _mm512_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]), bfloat16_to_float32(p0[A_hstep * 8]), bfloat16_to_float32(p0[A_hstep * 9]), bfloat16_to_float32(p0[A_hstep * 10]), bfloat16_to_float32(p0[A_hstep * 11]), bfloat16_to_float32(p0[A_hstep * 12]), bfloat16_to_float32(p0[A_hstep * 13]), bfloat16_to_float32(p0[A_hstep * 14]), bfloat16_to_float32(p0[A_hstep * 15]));
                        __m128i _q = float2int8_avx512(_mm512_mul_ps(_p, _scale512));
                        _mm_storeu_si128((__m128i*)pp, _q);
                        pp += 16;
                        p0 += A_hstep * 16;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        __m256i _q16 = _mm256_cvtepi8_epi16(_q);
                        __m256i _q32 = _mm256_madd_epi16(_q16, _mm256_set1_epi16(1));
                        w_shift += _mm_reduce_add_epi32(_mm256_castsi256_si128(_q32));
                        w_shift += _mm_reduce_add_epi32(_mm256_extracti128_si256(_q32, 1));
#endif
                    }
#endif // __AVX512F__
#if __AVX2__
                    __m256 _scale256 = _mm256_set1_ps(scale);
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _p = _mm256_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
                        const int64_t q = float2int8_avx(_mm256_mul_ps(_p, _scale256));
                        _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(q));
                        pp += 8;
                        p0 += A_hstep * 8;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
#if defined(__x86_64__) || defined(_M_X64)
                        __m128i _q8 = _mm_cvtsi64_si128(q);
#else
                        __m128i _q8 = _mm_loadl_epi64((const __m128i*)(pp - 8));
#endif
                        __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                        w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                    }
#endif // __AVX2__
#endif // __AVX__
                    __m128 _scale128 = _mm_set1_ps(scale);
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                        const int32_t q = float2int8_sse(_mm_mul_ps(_p, _scale128));
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(q)));
                        pp += 4;
                        p0 += A_hstep * 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                        __m128i _q8 = _mm_cvtsi32_si128(q);
                        __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                        w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                    }
#endif // __SSE2__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    if (max_kk0 >= 4)
                    {
                        _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(w_shift * 127)));
                        pp += 4;
                    }
#endif
                    for (; kk < max_kk0; kk++)
                    {
                        float v = bfloat16_to_float32(p0[0]);
                        *pp++ = float2int8(v * scale);
                        p0 += A_hstep;
                    }

                    pd++;
                }
            }
        }
        return;
    }
    const float* input_scale_ptr = (const float*)input_scales + k;

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const int block_count = (max_kk + block_size - 1) / block_size;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

            if (elempack == 16)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9, _p10, _p11, _p12, _p13, _p14, _p15;

                    _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 0 * 16)));
                    _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 1 * 16)));
                    _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 2 * 16)));
                    _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 3 * 16)));
                    _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 4 * 16)));
                    _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 5 * 16)));
                    _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 6 * 16)));
                    _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 7 * 16)));
                    _p8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 8 * 16)));
                    _p9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 9 * 16)));
                    _p10 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 10 * 16)));
                    _p11 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 11 * 16)));
                    _p12 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 12 * 16)));
                    _p13 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 13 * 16)));
                    _p14 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 14 * 16)));
                    _p15 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 15 * 16)));
                    transpose16x16_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9, _p10, _p11, _p12, _p13, _p14, _p15);

                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p0), _mm512_set1_ps(psa[0])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p1), _mm512_set1_ps(psa[1])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p2), _mm512_set1_ps(psa[2])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p3), _mm512_set1_ps(psa[3])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p4), _mm512_set1_ps(psa[4])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p5), _mm512_set1_ps(psa[5])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p6), _mm512_set1_ps(psa[6])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p7), _mm512_set1_ps(psa[7])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p8), _mm512_set1_ps(psa[8])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p9), _mm512_set1_ps(psa[9])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p10), _mm512_set1_ps(psa[10])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p11), _mm512_set1_ps(psa[11])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p12), _mm512_set1_ps(psa[12])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p13), _mm512_set1_ps(psa[13])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p14), _mm512_set1_ps(psa[14])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p15), _mm512_set1_ps(psa[15])));
                    p0a += A_hstep * 16;
                    psa += 16;
                }

                __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9, _p10, _p11, _p12, _p13, _p14, _p15;

                    _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 0 * 16)));
                    _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 1 * 16)));
                    _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 2 * 16)));
                    _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 3 * 16)));
                    _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 4 * 16)));
                    _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 5 * 16)));
                    _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 6 * 16)));
                    _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 7 * 16)));
                    _p8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 8 * 16)));
                    _p9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 9 * 16)));
                    _p10 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 10 * 16)));
                    _p11 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 11 * 16)));
                    _p12 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 12 * 16)));
                    _p13 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 13 * 16)));
                    _p14 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 14 * 16)));
                    _p15 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 15 * 16)));
                    transpose16x16_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9, _p10, _p11, _p12, _p13, _p14, _p15);

                    _p0 = _mm512_mul_ps(_mm512_mul_ps(_p0, _mm512_set1_ps(ps[0])), _scale);
                    _p1 = _mm512_mul_ps(_mm512_mul_ps(_p1, _mm512_set1_ps(ps[1])), _scale);
                    _p2 = _mm512_mul_ps(_mm512_mul_ps(_p2, _mm512_set1_ps(ps[2])), _scale);
                    _p3 = _mm512_mul_ps(_mm512_mul_ps(_p3, _mm512_set1_ps(ps[3])), _scale);
                    _p4 = _mm512_mul_ps(_mm512_mul_ps(_p4, _mm512_set1_ps(ps[4])), _scale);
                    _p5 = _mm512_mul_ps(_mm512_mul_ps(_p5, _mm512_set1_ps(ps[5])), _scale);
                    _p6 = _mm512_mul_ps(_mm512_mul_ps(_p6, _mm512_set1_ps(ps[6])), _scale);
                    _p7 = _mm512_mul_ps(_mm512_mul_ps(_p7, _mm512_set1_ps(ps[7])), _scale);
                    _p8 = _mm512_mul_ps(_mm512_mul_ps(_p8, _mm512_set1_ps(ps[8])), _scale);
                    _p9 = _mm512_mul_ps(_mm512_mul_ps(_p9, _mm512_set1_ps(ps[9])), _scale);
                    _p10 = _mm512_mul_ps(_mm512_mul_ps(_p10, _mm512_set1_ps(ps[10])), _scale);
                    _p11 = _mm512_mul_ps(_mm512_mul_ps(_p11, _mm512_set1_ps(ps[11])), _scale);
                    _p12 = _mm512_mul_ps(_mm512_mul_ps(_p12, _mm512_set1_ps(ps[12])), _scale);
                    _p13 = _mm512_mul_ps(_mm512_mul_ps(_p13, _mm512_set1_ps(ps[13])), _scale);
                    _p14 = _mm512_mul_ps(_mm512_mul_ps(_p14, _mm512_set1_ps(ps[14])), _scale);
                    _p15 = _mm512_mul_ps(_mm512_mul_ps(_p15, _mm512_set1_ps(ps[15])), _scale);
#if __AVX512VNNI__
    {
                        __m128i _q0 = float2int8_avx512(_p0);
                        __m128i _q1 = float2int8_avx512(_p1);
                        __m128i _q2 = float2int8_avx512(_p2);
                        __m128i _q3 = float2int8_avx512(_p3);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p4);
                        __m128i _q1 = float2int8_avx512(_p5);
                        __m128i _q2 = float2int8_avx512(_p6);
                        __m128i _q3 = float2int8_avx512(_p7);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p8);
                        __m128i _q1 = float2int8_avx512(_p9);
                        __m128i _q2 = float2int8_avx512(_p10);
                        __m128i _q3 = float2int8_avx512(_p11);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p12);
                        __m128i _q1 = float2int8_avx512(_p13);
                        __m128i _q2 = float2int8_avx512(_p14);
                        __m128i _q3 = float2int8_avx512(_p15);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                    }
#else

                    {
                        __m128i _q0 = float2int8_avx512(_p0);
                        __m128i _q1 = float2int8_avx512(_p1);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p2);
                        __m128i _q1 = float2int8_avx512(_p3);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p4);
                        __m128i _q1 = float2int8_avx512(_p5);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p6);
                        __m128i _q1 = float2int8_avx512(_p7);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p8);
                        __m128i _q1 = float2int8_avx512(_p9);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p10);
                        __m128i _q1 = float2int8_avx512(_p11);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p12);
                        __m128i _q1 = float2int8_avx512(_p13);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p14);
                        __m128i _q1 = float2int8_avx512(_p15);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
#endif
                    p0 += A_hstep * 16;
                    ps += 16;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm512_storeu_si512((__m512i*)pp, _w_shift);
                    pp += 64;
                }
#endif
                pd += 16;
            }
            if (elempack == 8)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    __m512 _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7;

                    _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 0 * 16)));
                    _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 1 * 16)));
                    _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 2 * 16)));
                    _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 3 * 16)));
                    _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 4 * 16)));
                    _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 5 * 16)));
                    _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 6 * 16)));
                    _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 7 * 16)));
                    transpose16x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p0), _mm512_set1_ps(psa[0])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p1), _mm512_set1_ps(psa[1])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p2), _mm512_set1_ps(psa[2])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p3), _mm512_set1_ps(psa[3])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p4), _mm512_set1_ps(psa[4])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p5), _mm512_set1_ps(psa[5])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p6), _mm512_set1_ps(psa[6])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p7), _mm512_set1_ps(psa[7])));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    __m512 _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7;

                    _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 0 * 16)));
                    _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 1 * 16)));
                    _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 2 * 16)));
                    _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 3 * 16)));
                    _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 4 * 16)));
                    _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 5 * 16)));
                    _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 6 * 16)));
                    _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 7 * 16)));
                    transpose16x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                    _p0 = _mm512_mul_ps(_mm512_mul_ps(_p0, _mm512_set1_ps(ps[0])), _scale);
                    _p1 = _mm512_mul_ps(_mm512_mul_ps(_p1, _mm512_set1_ps(ps[1])), _scale);
                    _p2 = _mm512_mul_ps(_mm512_mul_ps(_p2, _mm512_set1_ps(ps[2])), _scale);
                    _p3 = _mm512_mul_ps(_mm512_mul_ps(_p3, _mm512_set1_ps(ps[3])), _scale);
                    _p4 = _mm512_mul_ps(_mm512_mul_ps(_p4, _mm512_set1_ps(ps[4])), _scale);
                    _p5 = _mm512_mul_ps(_mm512_mul_ps(_p5, _mm512_set1_ps(ps[5])), _scale);
                    _p6 = _mm512_mul_ps(_mm512_mul_ps(_p6, _mm512_set1_ps(ps[6])), _scale);
                    _p7 = _mm512_mul_ps(_mm512_mul_ps(_p7, _mm512_set1_ps(ps[7])), _scale);
#if __AVX512VNNI__
    {
                        __m128i _q0 = float2int8_avx512(_p0);
                        __m128i _q1 = float2int8_avx512(_p1);
                        __m128i _q2 = float2int8_avx512(_p2);
                        __m128i _q3 = float2int8_avx512(_p3);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p4);
                        __m128i _q1 = float2int8_avx512(_p5);
                        __m128i _q2 = float2int8_avx512(_p6);
                        __m128i _q3 = float2int8_avx512(_p7);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                    }
#else

                    {
                        __m128i _q0 = float2int8_avx512(_p0);
                        __m128i _q1 = float2int8_avx512(_p1);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p2);
                        __m128i _q1 = float2int8_avx512(_p3);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p4);
                        __m128i _q1 = float2int8_avx512(_p5);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p6);
                        __m128i _q1 = float2int8_avx512(_p7);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
#endif
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm512_storeu_si512((__m512i*)pp, _w_shift);
                    pp += 64;
                }
#endif
                pd += 16;
            }
            if (elempack == 4)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    __m512 _p0, _p1, _p2, _p3;

                    _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 0 * 16)));
                    _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 1 * 16)));
                    _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 2 * 16)));
                    _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 3 * 16)));
                    transpose16x4_ps(_p0, _p1, _p2, _p3);

                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p0), _mm512_set1_ps(psa[0])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p1), _mm512_set1_ps(psa[1])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p2), _mm512_set1_ps(psa[2])));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p3), _mm512_set1_ps(psa[3])));
                    p0a += A_hstep * 4;
                    psa += 4;
                }

                __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                _mm512_storeu_ps(pd, _descale);
#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    __m512 _p0, _p1, _p2, _p3;

                    _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 0 * 16)));
                    _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 1 * 16)));
                    _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 2 * 16)));
                    _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 3 * 16)));
                    transpose16x4_ps(_p0, _p1, _p2, _p3);

                    _p0 = _mm512_mul_ps(_mm512_mul_ps(_p0, _mm512_set1_ps(ps[0])), _scale);
                    _p1 = _mm512_mul_ps(_mm512_mul_ps(_p1, _mm512_set1_ps(ps[1])), _scale);
                    _p2 = _mm512_mul_ps(_mm512_mul_ps(_p2, _mm512_set1_ps(ps[2])), _scale);
                    _p3 = _mm512_mul_ps(_mm512_mul_ps(_p3, _mm512_set1_ps(ps[3])), _scale);
#if __AVX512VNNI__
    {
                        __m128i _q0 = float2int8_avx512(_p0);
                        __m128i _q1 = float2int8_avx512(_p1);
                        __m128i _q2 = float2int8_avx512(_p2);
                        __m128i _q3 = float2int8_avx512(_p3);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                        _mm512_storeu_si512((__m512i*)pp, _q);
                        _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                        pp += 64;
                    }
#else

                    {
                        __m128i _q0 = float2int8_avx512(_p0);
                        __m128i _q1 = float2int8_avx512(_p1);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
                    {
                        __m128i _q0 = float2int8_avx512(_p2);
                        __m128i _q1 = float2int8_avx512(_p3);
                        _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                        _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                        pp += 32;
                    }
#endif
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm512_storeu_si512((__m512i*)pp, _w_shift);
                    pp += 64;
                }
#endif
                pd += 16;
            }

            if (elempack == 1)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;

                __m512 _absmax = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a)));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p), _mm512_set1_ps(psa[0])));
                    p0a += A_hstep;
                    psa++;
                }

                __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
                __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
                _mm512_storeu_ps(pd, _descale);

#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + A_hstep)));
                    __m512 _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + A_hstep * 2)));
                    __m512 _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + A_hstep * 3)));
                    _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(ps[0]));
                    _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(ps[1]));
                    _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(ps[2]));
                    _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(ps[3]));
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_p2, _scale));
                    __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_p3, _scale));
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __m512i _q = combine4x4_epi32(_q0, _q1, _q2, _q3);
                    _mm512_storeu_si512((__m512i*)pp, _q);
                    _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _q);
                    pp += 64;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#endif // __AVX512VNNI__
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm512_storeu_si512((__m512i*)pp, _w_shift);
                    pp += 64;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + A_hstep)));
                    _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(ps[0]));
                    _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(ps[1]));
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                    pp += 32;
                    p0 += A_hstep * 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                    _p = _mm512_mul_ps(_p, _mm512_set1_ps(ps[0]));
                    _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
                    pp += 16;
                    p0 += A_hstep;
                    ps++;
                }

                pd += 16;
            }
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    signed char* pp1 = pp + AT_tile.w * 4;
    float* pd1 = pd + AT_descales_tile.w * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __AVX512F__
            if (elempack == 16)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax0 = _mm512_setzero_ps();
                __m512 _absmax1 = _mm512_setzero_ps();
                __m512 _absmax2 = _mm512_setzero_ps();
                __m512 _absmax3 = _mm512_setzero_ps();
                __m512 _absmax4 = _mm512_setzero_ps();
                __m512 _absmax5 = _mm512_setzero_ps();
                __m512 _absmax6 = _mm512_setzero_ps();
                __m512 _absmax7 = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _s = _mm512_loadu_ps(psa);
                    __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 16)));
                    __m512 _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 32)));
                    __m512 _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 48)));
                    __m512 _p4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 64)));
                    __m512 _p5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 80)));
                    __m512 _p6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 96)));
                    __m512 _p7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 112)));
                    _absmax0 = _mm512_max_ps(_absmax0, _mm512_mul_ps(abs512_ps(_p0), _s));
                    _absmax1 = _mm512_max_ps(_absmax1, _mm512_mul_ps(abs512_ps(_p1), _s));
                    _absmax2 = _mm512_max_ps(_absmax2, _mm512_mul_ps(abs512_ps(_p2), _s));
                    _absmax3 = _mm512_max_ps(_absmax3, _mm512_mul_ps(abs512_ps(_p3), _s));
                    _absmax4 = _mm512_max_ps(_absmax4, _mm512_mul_ps(abs512_ps(_p4), _s));
                    _absmax5 = _mm512_max_ps(_absmax5, _mm512_mul_ps(abs512_ps(_p5), _s));
                    _absmax6 = _mm512_max_ps(_absmax6, _mm512_mul_ps(abs512_ps(_p6), _s));
                    _absmax7 = _mm512_max_ps(_absmax7, _mm512_mul_ps(abs512_ps(_p7), _s));
                    p0a += A_hstep * 16;
                    psa += 16;
                }

                __m256 _absmax = _mm256_setr_ps(_mm512_comp_reduce_max_ps(_absmax0), _mm512_comp_reduce_max_ps(_absmax1), _mm512_comp_reduce_max_ps(_absmax2), _mm512_comp_reduce_max_ps(_absmax3), _mm512_comp_reduce_max_ps(_absmax4), _mm512_comp_reduce_max_ps(_absmax5), _mm512_comp_reduce_max_ps(_absmax6), _mm512_comp_reduce_max_ps(_absmax7));
                __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
                _mm256_storeu_ps(pd, _descale);
                float scale[8];
                _mm256_storeu_ps(scale, _scale);
#if __AVX512VNNI__
                __m512i _w_shift = _mm512_setzero_si512();
                __m512i _v127 = _mm512_set1_epi8(127);
#endif
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _s = _mm512_loadu_ps(ps);
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _s), _mm512_set1_ps(scale[0])));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _s), _mm512_set1_ps(scale[1])));
                    __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 32))), _s), _mm512_set1_ps(scale[2])));
                    __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 48))), _s), _mm512_set1_ps(scale[3])));
                    __m128i _q4 = float2int8_avx512(_mm512_mul_ps(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 64))), _s), _mm512_set1_ps(scale[4])));
                    __m128i _q5 = float2int8_avx512(_mm512_mul_ps(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 80))), _s), _mm512_set1_ps(scale[5])));
                    __m128i _q6 = float2int8_avx512(_mm512_mul_ps(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 96))), _s), _mm512_set1_ps(scale[6])));
                    __m128i _q7 = float2int8_avx512(_mm512_mul_ps(_mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 112))), _s), _mm512_set1_ps(scale[7])));
#if __AVX512VNNI__
                    transpose4x8_epi32(_q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7);
#else
                    transpose8x8_epi16(_q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7);
#endif
                    __m512i _t0 = combine4x4_epi32(_q0, _q1, _q2, _q3);
                    __m512i _t1 = combine4x4_epi32(_q4, _q5, _q6, _q7);
#if __AVX512VNNI__
                    _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _t0);
                    _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _t1);
#endif
                    _mm512_storeu_si512((__m512i*)pp, _t0);
                    _mm512_storeu_si512((__m512i*)(pp + 64), _t1);
                    pp += 128;
                    p0 += A_hstep * 16;
                    ps += 16;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    __m256i _w_shift8 = _mm256_add_epi32(_mm512_extracti32x8_epi32(_w_shift, 0), _mm512_extracti32x8_epi32(_w_shift, 1));
                    _mm256_storeu_si256((__m256i*)pp, _w_shift8);
                    pp += 32;
                }
#endif

                pd += 8;
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                float absmax[8];
                for (int m = 0; m < 8; m++)
                {
                    const unsigned short* p0a = p0 + m * 8;
                    __m256 _absmax = _mm256_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m256 _p = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a)), _mm256_loadu_ps(ps + kk));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p));
                        p0a += A_hstep * 8;
                    }
                    absmax[m] = _mm256_reduce_max_ps(_absmax);
                }
                __m256 _descale = _mm256_div_ps(_mm256_loadu_ps(absmax), _mm256_set1_ps(127.f));
#if __AVX2__
                _mm256_storeu_ps(pd, _descale);
#else
                _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _w_shift = _mm256_setzero_si256();
#endif
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    __m256 _s = _mm256_loadu_ps(ps);
                    __m256 _p0 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _s), _mm256_set1_ps(absmax[0] == 0.f ? 0.f : 127.f / absmax[0]));
                    __m256 _p1 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _s), _mm256_set1_ps(absmax[1] == 0.f ? 0.f : 127.f / absmax[1]));
                    __m256 _p2 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 16))), _s), _mm256_set1_ps(absmax[2] == 0.f ? 0.f : 127.f / absmax[2]));
                    __m256 _p3 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 24))), _s), _mm256_set1_ps(absmax[3] == 0.f ? 0.f : 127.f / absmax[3]));
                    __m256 _p4 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 32))), _s), _mm256_set1_ps(absmax[4] == 0.f ? 0.f : 127.f / absmax[4]));
                    __m256 _p5 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 40))), _s), _mm256_set1_ps(absmax[5] == 0.f ? 0.f : 127.f / absmax[5]));
                    __m256 _p6 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 48))), _s), _mm256_set1_ps(absmax[6] == 0.f ? 0.f : 127.f / absmax[6]));
                    __m256 _p7 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 56))), _s), _mm256_set1_ps(absmax[7] == 0.f ? 0.f : 127.f / absmax[7]));
                    transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                    __m128i _q0 = float2int8_avx(_p0, _p2);
                    __m128i _q1 = float2int8_avx(_p1, _p3);
                    __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                    _q0 = _mm_unpacklo_epi16(_q01, _q23);
                    _q1 = _mm_unpackhi_epi16(_q01, _q23);
                    __m256i _q = combine4x2_epi32(_q0, _q1);
                    _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q01);
                    _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                    _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif
#if __AVX2__
                    pp += 32;
#else
                    pp += 16;
                    pp1 += 16;
#endif
                    _q0 = float2int8_avx(_p4, _p6);
                    _q1 = float2int8_avx(_p5, _p7);
                    _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                    _q0 = _mm_unpacklo_epi16(_q01, _q23);
                    _q1 = _mm_unpackhi_epi16(_q01, _q23);
                    _q = combine4x2_epi32(_q0, _q1);
                    _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q01);
                    _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                    _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif
#if __AVX2__
                    pp += 32;
#else
                    pp += 16;
                    pp1 += 16;
#endif
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm256_storeu_si256((__m256i*)pp, _w_shift);
                    pp += 32;
                }
#endif
#if __AVX2__
                pd += 8;
#else
                pd += 4;
                pd1 += 4;
#endif
            }
            if (elempack == 4)
            {
                float absmax[8];
                for (int m = 0; m < 8; m++)
                {
                    const unsigned short* p0a = p0 + m * 4;
                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _p = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a)), _mm_loadu_ps(ps + kk));
                        _absmax = _mm_max_ps(_absmax, abs_ps(_p));
                        p0a += A_hstep * 4;
                    }
                    absmax[m] = _mm_reduce_max_ps(_absmax);
                }
                __m256 _descale = _mm256_div_ps(_mm256_loadu_ps(absmax), _mm256_set1_ps(127.f));
#if __AVX2__
                _mm256_storeu_ps(pd, _descale);
#else
                _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif

                __m256 _scale0 = combine4x2_ps(_mm_set1_ps(absmax[0] == 0.f ? 0.f : 127.f / absmax[0]), _mm_set1_ps(absmax[1] == 0.f ? 0.f : 127.f / absmax[1]));
                __m256 _scale1 = combine4x2_ps(_mm_set1_ps(absmax[2] == 0.f ? 0.f : 127.f / absmax[2]), _mm_set1_ps(absmax[3] == 0.f ? 0.f : 127.f / absmax[3]));
                __m256 _scale2 = combine4x2_ps(_mm_set1_ps(absmax[4] == 0.f ? 0.f : 127.f / absmax[4]), _mm_set1_ps(absmax[5] == 0.f ? 0.f : 127.f / absmax[5]));
                __m256 _scale3 = combine4x2_ps(_mm_set1_ps(absmax[6] == 0.f ? 0.f : 127.f / absmax[6]), _mm_set1_ps(absmax[7] == 0.f ? 0.f : 127.f / absmax[7]));
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _w_shift = _mm256_setzero_si256();
#endif
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    __m256 _s = combine4x2_ps(_mm_loadu_ps(ps), _mm_loadu_ps(ps));
                    __m256 _p0 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _s), _scale0);
                    __m256 _p1 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _s), _scale1);
                    __m256 _p2 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 16))), _s), _scale2);
                    __m256 _p3 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 24))), _s), _scale3);
                    __m128i _q0 = float2int8_avx(_p0, _p1);
                    __m128i _q1 = float2int8_avx(_p2, _p3);
#if __AVX512VNNI__ || __AVXVNNI__
                    __m256i _q = combine4x2_epi32(_q0, _q1);
                    _mm256_storeu_si256((__m256i*)pp, _q);
#if !__AVXVNNIINT8__
                    _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
                    pp += 32;
#else
#if __AVX2__
                    __m128i _t0 = _mm_unpacklo_epi16(_q0, _q1);
                    __m128i _t1 = _mm_unpackhi_epi16(_q0, _q1);
                    __m128i _t2 = _mm_unpacklo_epi16(_t0, _t1);
                    __m128i _t3 = _mm_unpackhi_epi16(_t0, _t1);
                    _t0 = _mm_unpacklo_epi16(_t2, _t3);
                    _t1 = _mm_unpackhi_epi16(_t2, _t3);
                    _mm_storeu_si128((__m128i*)pp, _t0);
                    _mm_storeu_si128((__m128i*)(pp + 16), _t1);
                    pp += 32;
#else
                    __m128i _si = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                    _mm_storeu_si128((__m128i*)pp, _mm_shuffle_epi8(_q0, _si));
                    _mm_storeu_si128((__m128i*)pp1, _mm_shuffle_epi8(_q1, _si));
                    pp += 16;
                    pp1 += 16;
#endif
#endif
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm256_storeu_si256((__m256i*)pp, _w_shift);
                    pp += 32;
                }
#endif
#if __AVX2__
                pd += 8;
#else
                pd += 4;
                pd1 += 4;
#endif
            }
            if (elempack == 1)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;

                __m256 _absmax = _mm256_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a)));
                    _absmax = _mm256_max_ps(_absmax, _mm256_mul_ps(abs256_ps(_p), _mm256_set1_ps(psa[0])));
                    p0a += A_hstep;
                    psa++;
                }

                __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
                __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
                __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
#if __AVX2__
                _mm256_storeu_ps(pd, _descale);
#else
                _mm_storeu_ps(pd, _mm256_castps256_ps128(_descale));
                _mm_storeu_ps(pd1, _mm256_extractf128_ps(_descale, 1));
#endif
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _w_shift = _mm256_setzero_si256();
#endif
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep)));
                    __m256 _p2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2)));
                    __m256 _p3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3)));
                    _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(ps[0]));
                    _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(ps[1]));
                    _p2 = _mm256_mul_ps(_p2, _mm256_set1_ps(ps[2]));
                    _p3 = _mm256_mul_ps(_p3, _mm256_set1_ps(ps[3]));
                    _p0 = _mm256_mul_ps(_p0, _scale);
                    _p1 = _mm256_mul_ps(_p1, _scale);
                    _p2 = _mm256_mul_ps(_p2, _scale);
                    _p3 = _mm256_mul_ps(_p3, _scale);

                    __m128i _q0 = float2int8_avx(_p0, _p2);
                    __m128i _q1 = float2int8_avx(_p1, _p3);
                    __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    __m128i _q23 = _mm_unpackhi_epi8(_q0, _q1);
#if __AVX512VNNI__ || __AVXVNNI__
                    _q0 = _mm_unpacklo_epi16(_q01, _q23);
                    _q1 = _mm_unpackhi_epi16(_q01, _q23);
                    __m256i _q = combine4x2_epi32(_q0, _q1);
                    _mm256_storeu_si256((__m256i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _mm256_set1_epi8(127), _q);
#endif
#elif __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q01);
                    _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
                    _mm_storeu_si128((__m128i*)pp1, _mm_unpackhi_epi64(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX2__
                    pp += 32;
#else
                    pp += 16;
                    pp1 += 16;
#endif
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm256_storeu_si256((__m256i*)pp, _w_shift);
                    pp += 32;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256 _p0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + A_hstep)));
                    _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(ps[0]));
                    _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(ps[1]));
                    _p0 = _mm256_mul_ps(_p0, _scale);
                    _p1 = _mm256_mul_ps(_p1, _scale);
                    __m128i _q = float2int8_avx(_p0, _p1);
                    __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                    _q = _mm_shuffle_epi8(_q, _si);
#if __AVX2__
                    _mm_storeu_si128((__m128i*)pp, _q);
                    pp += 16;
#else
                    _mm_storel_pd((double*)pp, _mm_castsi128_pd(_q));
                    _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_q));
                    pp += 8;
                    pp1 += 8;
#endif
                    p0 += A_hstep * 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                    _p = _mm256_mul_ps(_p, _mm256_set1_ps(ps[0]));
#if __AVX2__
                    _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(float2int8_avx(_mm256_mul_ps(_p, _scale))));
                    pp += 8;
#else
                    const uint64_t q = (uint64_t)float2int8_avx(_mm256_mul_ps(_p, _scale));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128((int)q)));
                    _mm_store_ss((float*)pp1, _mm_castsi128_ps(_mm_cvtsi32_si128((int)(q >> 32))));
                    pp += 4;
                    pp1 += 4;
#endif
                    p0 += A_hstep;
                    ps++;
                }

#if __AVX2__
                pd += 8;
#else
                pd += 4;
                pd1 += 4;
#endif
            }
        }
#if !__AVX2__
        pp = pp1;
        pp1 = pp + AT_tile.w * 4;
        pd = pd1;
        pd1 = pd + AT_descales_tile.w * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax0 = _mm512_setzero_ps();
                __m512 _absmax1 = _mm512_setzero_ps();
                __m512 _absmax2 = _mm512_setzero_ps();
                __m512 _absmax3 = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _s = _mm512_loadu_ps(psa);
                    __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 16)));
                    __m512 _p2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 32)));
                    __m512 _p3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 48)));
                    _absmax0 = _mm512_max_ps(_absmax0, _mm512_mul_ps(abs512_ps(_p0), _s));
                    _absmax1 = _mm512_max_ps(_absmax1, _mm512_mul_ps(abs512_ps(_p1), _s));
                    _absmax2 = _mm512_max_ps(_absmax2, _mm512_mul_ps(abs512_ps(_p2), _s));
                    _absmax3 = _mm512_max_ps(_absmax3, _mm512_mul_ps(abs512_ps(_p3), _s));
                    p0a += A_hstep * 16;
                    psa += 16;
                }

                float absmax0 = _mm512_comp_reduce_max_ps(_absmax0);
                float absmax1 = _mm512_comp_reduce_max_ps(_absmax1);
                float absmax2 = _mm512_comp_reduce_max_ps(_absmax2);
                float absmax3 = _mm512_comp_reduce_max_ps(_absmax3);
                __m128 _absmax = _mm_setr_ps(absmax0, absmax1, absmax2, absmax3);
                __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                _mm_storeu_ps(pd, _descale);
                __m512 _scale0 = _mm512_set1_ps(_mm_cvtss_f32(_scale));
                __m512 _scale1 = _mm512_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(1, 1, 1, 1))));
                __m512 _scale2 = _mm512_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(2, 2, 2, 2))));
                __m512 _scale3 = _mm512_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(_scale, _scale, _MM_SHUFFLE(3, 3, 3, 3))));
#if __AVX512VNNI__
                __m128i _w_shift = _mm_setzero_si128();
                __m128i _v127 = _mm_set1_epi8(127);
#endif
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _s = _mm512_loadu_ps(ps);
                    __m512 _p0 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _s);
                    __m512 _p1 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _s);
                    __m512 _p2 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 32))), _s);
                    __m512 _p3 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 48))), _s);
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale0));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale1));
                    __m128i _q2 = float2int8_avx512(_mm512_mul_ps(_p2, _scale2));
                    __m128i _q3 = float2int8_avx512(_mm512_mul_ps(_p3, _scale3));
#if __AVX512VNNI__
                    transpose4x4_epi32(_q0, _q1, _q2, _q3);
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _q0);
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _q1);
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _q2);
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _q3);
#else
                    transpose8x4_epi16(_q0, _q1, _q2, _q3);
#endif
                    _mm_storeu_si128((__m128i*)pp, _q0);
                    _mm_storeu_si128((__m128i*)(pp + 16), _q1);
                    _mm_storeu_si128((__m128i*)(pp + 32), _q2);
                    _mm_storeu_si128((__m128i*)(pp + 48), _q3);
                    pp += 64;
                    p0 += A_hstep * 16;
                    ps += 16;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _mm_storeu_si128((__m128i*)pp, _w_shift);
                    pp += 16;
                }
#endif

                pd += 4;
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                float absmax[4];
                for (int m = 0; m < 4; m++)
                {
                    const unsigned short* p0a = p0 + m * 8;
                    __m256 _absmax = _mm256_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m256 _p = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a)), _mm256_loadu_ps(ps + kk));
                        _absmax = _mm256_max_ps(_absmax, abs256_ps(_p));
                        p0a += A_hstep * 8;
                    }
                    absmax[m] = _mm256_reduce_max_ps(_absmax);
                    pd[m] = absmax[m] / 127.f;
                }
                __m256 _scale0 = _mm256_set1_ps(absmax[0] == 0.f ? 0.f : 127.f / absmax[0]);
                __m256 _scale1 = _mm256_set1_ps(absmax[1] == 0.f ? 0.f : 127.f / absmax[1]);
                __m256 _scale2 = _mm256_set1_ps(absmax[2] == 0.f ? 0.f : 127.f / absmax[2]);
                __m256 _scale3 = _mm256_set1_ps(absmax[3] == 0.f ? 0.f : 127.f / absmax[3]);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _w_shift = _mm_setzero_si128();
#endif
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    __m256 _s = _mm256_loadu_ps(ps);
                    __m256 _p0 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _s), _scale0);
                    __m256 _p1 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _s), _scale1);
                    __m256 _p2 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 16))), _s), _scale2);
                    __m256 _p3 = _mm256_mul_ps(_mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 24))), _s), _scale3);
                    __m128i _q0 = float2int8_avx(_p0, _p2);
                    __m128i _q1 = float2int8_avx(_p1, _p3);
#if __AVX512VNNI__ || __AVXVNNI__
                    __m128i _t0 = _mm_unpacklo_epi32(_q0, _q1);
                    __m128i _t1 = _mm_unpackhi_epi32(_q0, _q1);
                    _q0 = _mm_unpacklo_epi64(_t0, _t1);
                    _q1 = _mm_unpackhi_epi64(_t0, _t1);
#if !__AVXVNNIINT8__
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q0);
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q1);
#endif
#else
                    __m128i _t0 = _mm_unpacklo_epi16(_q0, _q1);
                    __m128i _t1 = _mm_unpackhi_epi16(_q0, _q1);
                    _q0 = _mm_unpacklo_epi32(_t0, _t1);
                    _q1 = _mm_unpackhi_epi32(_t0, _t1);
#endif
                    _mm_storeu_si128((__m128i*)pp, _q0);
                    _mm_storeu_si128((__m128i*)(pp + 16), _q1);
                    pp += 32;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_storeu_si128((__m128i*)pp, _w_shift);
                    pp += 16;
                }
#endif
                pd += 4;
            }
#endif // __AVX__
            if (elempack == 4)
            {
                float absmax[4];
                for (int m = 0; m < 4; m++)
                {
                    const unsigned short* p0a = p0 + m * 4;
                    __m128 _absmax = _mm_setzero_ps();
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _p = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a)), _mm_loadu_ps(ps + kk));
                        _absmax = _mm_max_ps(_absmax, abs_ps(_p));
                        p0a += A_hstep * 4;
                    }
                    absmax[m] = _mm_reduce_max_ps(_absmax);
                    pd[m] = absmax[m] / 127.f;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _w_shift = _mm_setzero_si128();
#endif
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    __m128 _s = _mm_loadu_ps(ps);
                    __m128 _p0 = _mm_mul_ps(_mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), _s), _mm_set1_ps(absmax[0] == 0.f ? 0.f : 127.f / absmax[0]));
                    __m128 _p1 = _mm_mul_ps(_mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))), _s), _mm_set1_ps(absmax[1] == 0.f ? 0.f : 127.f / absmax[1]));
                    __m128 _p2 = _mm_mul_ps(_mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 8))), _s), _mm_set1_ps(absmax[2] == 0.f ? 0.f : 127.f / absmax[2]));
                    __m128 _p3 = _mm_mul_ps(_mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 12))), _s), _mm_set1_ps(absmax[3] == 0.f ? 0.f : 127.f / absmax[3]));
                    _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_p0));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_p1));
                    __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_p2));
                    __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_p3));
                    __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                    __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                    _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif
                    pp += 16;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_storeu_si128((__m128i*)pp, _w_shift);
                    pp += 16;
                }
#endif
                pd += 4;
            }
            if (elempack == 1)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;

                __m128 _absmax = _mm_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a)));
                    _absmax = _mm_max_ps(_absmax, _mm_mul_ps(abs_ps(_p), _mm_set1_ps(psa[0])));
                    p0a += A_hstep;
                    psa++;
                }

                __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
                __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
                __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
                __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
                _mm_storeu_ps(pd, _descale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _w_shift = _mm_setzero_si128();
#endif
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                    __m128 _p2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                    __m128 _p3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(ps[0]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(ps[1]));
                    _p2 = _mm_mul_ps(_p2, _mm_set1_ps(ps[2]));
                    _p3 = _mm_mul_ps(_p3, _mm_set1_ps(ps[3]));
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                    __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p2, _scale)));
                    __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p3, _scale)));
                    __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                    __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                    _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                    _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi64(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
                    pp += 16;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_storeu_si128((__m128i*)pp, _w_shift);
                    pp += 16;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(ps[0]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(ps[1]));
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                    _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                    pp += 8;
                    p0 += A_hstep * 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                    _p = _mm_mul_ps(_p, _mm_set1_ps(ps[0]));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p, _scale)))));
                    pp += 4;
                    p0 += A_hstep;
                    ps++;
                }

                pd += 4;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax0 = _mm512_setzero_ps();
                __m512 _absmax1 = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _s = _mm512_loadu_ps(psa);
                    __m512 _p0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0a + 16)));
                    _absmax0 = _mm512_max_ps(_absmax0, _mm512_mul_ps(abs512_ps(_p0), _s));
                    _absmax1 = _mm512_max_ps(_absmax1, _mm512_mul_ps(abs512_ps(_p1), _s));
                    p0a += A_hstep * 16;
                    psa += 16;
                }

                float absmax0 = _mm512_comp_reduce_max_ps(_absmax0);
                float absmax1 = _mm512_comp_reduce_max_ps(_absmax1);
                float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                __m512 _scale0 = _mm512_set1_ps(scale0);
                __m512 _scale1 = _mm512_set1_ps(scale1);
#if __AVX512VNNI__
                __m128i _w_shift = _mm_setzero_si128();
                __m128i _v127 = _mm_set1_epi8(127);
#endif
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _s = _mm512_loadu_ps(ps);
                    __m512 _p0 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)), _s);
                    __m512 _p1 = _mm512_mul_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16))), _s);
                    __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale0));
                    __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale1));
#if __AVX512VNNI__
                    __m128i _t0 = _mm_unpacklo_epi32(_q0, _q1);
                    __m128i _t1 = _mm_unpackhi_epi32(_q0, _q1);
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _t0);
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _t1);
#else
                    __m128i _t0 = _mm_unpacklo_epi16(_q0, _q1);
                    __m128i _t1 = _mm_unpackhi_epi16(_q0, _q1);
#endif
                    _mm_storeu_si128((__m128i*)pp, _t0);
                    _mm_storeu_si128((__m128i*)(pp + 16), _t1);
                    pp += 32;
                    p0 += A_hstep * 16;
                    ps += 16;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    _w_shift = _mm_shuffle_epi32(_w_shift, _MM_SHUFFLE(3, 1, 2, 0));
                    _w_shift = _mm_hadd_epi32(_w_shift, _w_shift);
                    _mm_storel_epi64((__m128i*)pp, _w_shift);
                    pp += 8;
                }
#endif

                pd += 2;
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m256 _absmax0 = _mm256_setzero_ps();
                __m256 _absmax1 = _mm256_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    __m256 _s = _mm256_loadu_ps(psa);
                    _absmax0 = _mm256_max_ps(_absmax0, _mm256_mul_ps(abs256_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a))), _s));
                    _absmax1 = _mm256_max_ps(_absmax1, _mm256_mul_ps(abs256_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0a + 8)))), _s));
                    p0a += A_hstep * 8;
                    psa += 8;
                }
                float absmax0 = _mm256_reduce_max_ps(_absmax0);
                float absmax1 = _mm256_reduce_max_ps(_absmax1);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                __m256 _scale0 = _mm256_set1_ps(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                __m256 _scale1 = _mm256_set1_ps(absmax1 == 0.f ? 0.f : 127.f / absmax1);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _w_shift = _mm_setzero_si128();
                __m128i _v127 = _mm_set1_epi8(127);
#endif
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    __m256 _s = _mm256_loadu_ps(ps);
                    __m256 _p0 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _s);
                    __m256 _p1 = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8))), _s);
                    __m128i _q = float2int8_avx(_mm256_mul_ps(_p0, _scale0), _mm256_mul_ps(_p1, _scale1));
                    _q = _mm_shuffle_epi32(_q, _MM_SHUFFLE(3, 1, 2, 0));
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _q);
#endif
#else
                    _q = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_q, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
#endif
                    _mm_storeu_si128((__m128i*)pp, _q);
                    pp += 16;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _w_shift = _mm_shuffle_epi32(_w_shift, _MM_SHUFFLE(3, 1, 2, 0));
                    _w_shift = _mm_hadd_epi32(_w_shift, _w_shift);
                    _mm_storel_epi64((__m128i*)pp, _w_shift);
                    pp += 8;
                }
#endif
                pd += 2;
            }
#endif // __AVX__
            if (elempack == 4)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m128 _absmax0 = _mm_setzero_ps();
                __m128 _absmax1 = _mm_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    __m128 _s = _mm_loadu_ps(psa);
                    _absmax0 = _mm_max_ps(_absmax0, _mm_mul_ps(abs_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a))), _s));
                    _absmax1 = _mm_max_ps(_absmax1, _mm_mul_ps(abs_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0a + 4)))), _s));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
                float absmax0 = _mm_reduce_max_ps(_absmax0);
                float absmax1 = _mm_reduce_max_ps(_absmax1);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                __m128 _scale0 = _mm_set1_ps(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                __m128 _scale1 = _mm_set1_ps(absmax1 == 0.f ? 0.f : 127.f / absmax1);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                int w_shift0 = 0;
                int w_shift1 = 0;
#endif
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    __m128 _s = _mm_loadu_ps(ps);
                    __m128 _p0 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), _s);
                    __m128 _p1 = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))), _s);
#if __AVX512VNNI__ || __AVXVNNI__
                    int64_t v = float2int8_sse(_mm_mul_ps(_p0, _scale0), _mm_mul_ps(_p1, _scale1));
                    *(int64_t*)pp = v;
#if !__AVXVNNIINT8__
                    w_shift0 += pp[0] + pp[1] + pp[2] + pp[3];
                    w_shift1 += pp[4] + pp[5] + pp[6] + pp[7];
#endif
#else
                    __m128 _t0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_mm_mul_ps(_p0, _scale0)), _mm_castps_pd(_mm_mul_ps(_p1, _scale1))));
                    __m128 _t1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_mm_mul_ps(_p0, _scale0)), _mm_castps_pd(_mm_mul_ps(_p1, _scale1))));
                    *(int64_t*)pp = float2int8_sse(_t0, _t1);
#endif
                    pp += 8;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    ((int*)pp)[0] = w_shift0 * 127;
                    ((int*)pp)[1] = w_shift1 * 127;
                    pp += 8;
                }
#endif
                pd += 2;
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;

                float absmax0 = 0.f;
                float absmax1 = 0.f;
                int kk = 0;
#if __SSE2__
                __m128 _absmax = _mm_setzero_ps();
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)p0a)));
                    _absmax = _mm_max_ps(_absmax, _mm_mul_ps(abs_ps(_p), _mm_set1_ps(psa[0])));
                    p0a += A_hstep;
                    psa++;
                }

                absmax0 = _mm_cvtss_f32(_absmax);
                absmax1 = _mm_cvtss_f32(_mm_shuffle_ps(_absmax, _absmax, _MM_SHUFFLE(1, 1, 1, 1)));
#endif // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    const float s = psa[0];
                    absmax0 = std::max(absmax0, (float)fabsf(bfloat16_to_float32(p0a[0])) * s);
                    absmax1 = std::max(absmax1, (float)fabsf(bfloat16_to_float32(p0a[1])) * s);
                    p0a += A_hstep;
                    psa++;
                }

                float scale0 = 0.f;
                float scale1 = 0.f;
                if (absmax0 != 0.f)
                    scale0 = 127.f / absmax0;
                if (absmax1 != 0.f)
                    scale1 = 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                kk = 0;
#if __SSE2__
                __m128 _scale = _mm_setr_ps(scale0, scale1, 0.f, 0.f);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _w_shift = _mm_setzero_si128();
#endif
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)p0)));
                    __m128 _p1 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)(p0 + A_hstep))));
                    __m128 _p2 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)(p0 + A_hstep * 2))));
                    __m128 _p3 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)(p0 + A_hstep * 3))));
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(ps[0]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(ps[1]));
                    _p2 = _mm_mul_ps(_p2, _mm_set1_ps(ps[2]));
                    _p3 = _mm_mul_ps(_p3, _mm_set1_ps(ps[3]));
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                    __m128i _q2 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p2, _scale)));
                    __m128i _q3 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p3, _scale)));
                    __m128i _q01 = _mm_unpacklo_epi8(_q0, _q1);
                    __m128i _q23 = _mm_unpacklo_epi8(_q2, _q3);
#if __AVX512VNNI__ || __AVXVNNI__
                    __m128i _q = _mm_unpacklo_epi16(_q01, _q23);
                    _mm_storel_epi64((__m128i*)pp, _q);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _mm_set1_epi8(127), _q);
#endif
#else
                    _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi32(_q01, _q23));
#endif // __AVX512VNNI__ || __AVXVNNI__
                    pp += 8;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_storel_epi64((__m128i*)pp, _w_shift);
                    pp += 8;
                }
#endif
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128 _p0 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)p0)));
                    __m128 _p1 = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)(p0 + A_hstep))));
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(ps[0]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(ps[1]));
                    __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                    __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_unpacklo_epi8(_q0, _q1)));
                    pp += 4;
                    p0 += A_hstep * 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = bfloat2float_sse(_mm_castps_si128(_mm_load_ss((const float*)p0)));
                    _p = _mm_mul_ps(_p, _mm_set1_ps(ps[0]));
                    unsigned int q = (unsigned int)float2int8_sse(_mm_mul_ps(_p, _scale));
                    pp[0] = (signed char)q;
                    pp[1] = (signed char)(q >> 8);
                    pp += 2;
                    p0 += A_hstep;
                    ps++;
                }
#endif // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[1]);
                    const float s = ps[0];
                    v0 *= s;
                    v1 *= s;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                    p0 += A_hstep;
                    ps++;
                }

                pd += 2;
            }
        }
    }

    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m512 _absmax = _mm512_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0a));
                    _absmax = _mm512_max_ps(_absmax, _mm512_mul_ps(abs512_ps(_p), _mm512_loadu_ps(psa)));
                    p0a += A_hstep * 16;
                    psa += 16;
                }

                float absmax = _mm512_comp_reduce_max_ps(_absmax);
                float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                pd[0] = absmax / 127.f;
                __m512 _scale = _mm512_set1_ps(scale);
#if __AVX512VNNI__
                int w_shift = 0;
#endif
                for (int kk = 0; kk < max_kk0; kk += 16)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                    _p = _mm512_mul_ps(_p, _mm512_loadu_ps(ps));
                    __m128i _q = float2int8_avx512(_mm512_mul_ps(_p, _scale));
                    _mm_storeu_si128((__m128i*)pp, _q);
#if __AVX512VNNI__
                    __m256i _q16 = _mm256_cvtepi8_epi16(_q);
                    __m256i _q32 = _mm256_madd_epi16(_q16, _mm256_set1_epi16(1));
                    w_shift += _mm_reduce_add_epi32(_mm256_castsi256_si128(_q32));
                    w_shift += _mm_reduce_add_epi32(_mm256_extracti128_si256(_q32, 1));
#endif
                    pp += 16;
                    p0 += A_hstep * 16;
                    ps += 16;
                }
#if __AVX512VNNI__
                if (max_kk0 >= 4)
                {
                    ((int*)pp)[0] = w_shift * 127;
                    pp += 4;
                }
#endif
                pd++;
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m256 _absmax = _mm256_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    _absmax = _mm256_max_ps(_absmax, _mm256_mul_ps(abs256_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0a))), _mm256_loadu_ps(psa)));
                    p0a += A_hstep * 8;
                    psa += 8;
                }
                float absmax = _mm256_reduce_max_ps(_absmax);
                pd[0] = absmax / 127.f;
                __m256 _scale = _mm256_set1_ps(absmax == 0.f ? 0.f : 127.f / absmax);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                int w_shift = 0;
#endif
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    __m256 _p = _mm256_mul_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)), _mm256_loadu_ps(ps));
                    const int64_t q = float2int8_avx(_mm256_mul_ps(_p, _scale));
                    _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(q));
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _q8 = _mm_cvtsi64_si128(q);
                    __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                    w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                    pp += 8;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    ((int*)pp)[0] = w_shift * 127;
                    pp += 4;
                }
#endif
                pd++;
            }
#endif // __AVX__
            if (elempack == 4)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                __m128 _absmax = _mm_setzero_ps();
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    _absmax = _mm_max_ps(_absmax, _mm_mul_ps(abs_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0a))), _mm_loadu_ps(psa)));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
                float absmax = _mm_reduce_max_ps(_absmax);
                pd[0] = absmax / 127.f;
                __m128 _scale = _mm_set1_ps(absmax == 0.f ? 0.f : 127.f / absmax);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                int w_shift = 0;
#endif
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    __m128 _p = _mm_mul_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)), _mm_loadu_ps(ps));
                    const int32_t q = float2int8_sse(_mm_mul_ps(_p, _scale));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(q)));
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _q8 = _mm_cvtsi32_si128(q);
                    __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                    w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                    pp += 4;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    ((int*)pp)[0] = w_shift * 127;
                    pp += 4;
                }
#endif
                pd++;
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                const unsigned short* p0a = p0;
                const float* psa = ps;
                float absmax = 0.f;
                int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _absmax512 = _mm512_setzero_ps();
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    __m512 _p = _mm512_setr_ps(bfloat16_to_float32(p0a[0]), bfloat16_to_float32(p0a[A_hstep]), bfloat16_to_float32(p0a[A_hstep * 2]), bfloat16_to_float32(p0a[A_hstep * 3]), bfloat16_to_float32(p0a[A_hstep * 4]), bfloat16_to_float32(p0a[A_hstep * 5]), bfloat16_to_float32(p0a[A_hstep * 6]), bfloat16_to_float32(p0a[A_hstep * 7]), bfloat16_to_float32(p0a[A_hstep * 8]), bfloat16_to_float32(p0a[A_hstep * 9]), bfloat16_to_float32(p0a[A_hstep * 10]), bfloat16_to_float32(p0a[A_hstep * 11]), bfloat16_to_float32(p0a[A_hstep * 12]), bfloat16_to_float32(p0a[A_hstep * 13]), bfloat16_to_float32(p0a[A_hstep * 14]), bfloat16_to_float32(p0a[A_hstep * 15]));
                    _absmax512 = _mm512_max_ps(_absmax512, _mm512_mul_ps(abs512_ps(_p), _mm512_loadu_ps(psa)));
                    p0a += A_hstep * 16;
                    psa += 16;
                }
                absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax512));
#endif // __AVX512F__
#if __AVX2__
                __m256 _absmax256 = _mm256_setzero_ps();
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _p = _mm256_setr_ps(bfloat16_to_float32(p0a[0]), bfloat16_to_float32(p0a[A_hstep]), bfloat16_to_float32(p0a[A_hstep * 2]), bfloat16_to_float32(p0a[A_hstep * 3]), bfloat16_to_float32(p0a[A_hstep * 4]), bfloat16_to_float32(p0a[A_hstep * 5]), bfloat16_to_float32(p0a[A_hstep * 6]), bfloat16_to_float32(p0a[A_hstep * 7]));
                    _absmax256 = _mm256_max_ps(_absmax256, _mm256_mul_ps(abs256_ps(_p), _mm256_loadu_ps(psa)));
                    p0a += A_hstep * 8;
                    psa += 8;
                }
                absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax256));
#endif // __AVX2__
#endif // __AVX__
                __m128 _absmax128 = _mm_setzero_ps();
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p = _mm_setr_ps(bfloat16_to_float32(p0a[0]), bfloat16_to_float32(p0a[A_hstep]), bfloat16_to_float32(p0a[A_hstep * 2]), bfloat16_to_float32(p0a[A_hstep * 3]));
                    _absmax128 = _mm_max_ps(_absmax128, _mm_mul_ps(abs_ps(_p), _mm_loadu_ps(psa)));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
                absmax = std::max(absmax, _mm_reduce_max_ps(_absmax128));
#endif // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    absmax = std::max(absmax, fabsf(bfloat16_to_float32(p0a[0])) * psa[0]);
                    p0a += A_hstep;
                    psa++;
                }

                if (absmax == 0.f)
                {
                    pd[0] = 0.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    memset(pp, 0, max_kk0 >= 4 ? max_kk0 + 4 : max_kk0);
                    pp += max_kk0 + (max_kk0 >= 4 ? 4 : 0);
#else
                    memset(pp, 0, max_kk0);
                    pp += max_kk0;
#endif
                    p0 += max_kk0 * A_hstep;
                    ps += max_kk0;
                    pd++;
                    continue;
                }

                const float scale = 127.f / absmax;
                pd[0] = absmax / 127.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                int w_shift = 0;
#endif
                kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _scale512 = _mm512_set1_ps(scale);
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    __m512 _p = _mm512_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]), bfloat16_to_float32(p0[A_hstep * 8]), bfloat16_to_float32(p0[A_hstep * 9]), bfloat16_to_float32(p0[A_hstep * 10]), bfloat16_to_float32(p0[A_hstep * 11]), bfloat16_to_float32(p0[A_hstep * 12]), bfloat16_to_float32(p0[A_hstep * 13]), bfloat16_to_float32(p0[A_hstep * 14]), bfloat16_to_float32(p0[A_hstep * 15]));
                    _p = _mm512_mul_ps(_p, _mm512_loadu_ps(ps));
                    __m128i _q = float2int8_avx512(_mm512_mul_ps(_p, _scale512));
                    _mm_storeu_si128((__m128i*)pp, _q);
                    pp += 16;
                    p0 += A_hstep * 16;
                    ps += 16;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m256i _q16 = _mm256_cvtepi8_epi16(_q);
                    __m256i _q32 = _mm256_madd_epi16(_q16, _mm256_set1_epi16(1));
                    w_shift += _mm_reduce_add_epi32(_mm256_castsi256_si128(_q32));
                    w_shift += _mm_reduce_add_epi32(_mm256_extracti128_si256(_q32, 1));
#endif
                }
#endif // __AVX512F__
#if __AVX2__
                __m256 _scale256 = _mm256_set1_ps(scale);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _p = _mm256_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]), bfloat16_to_float32(p0[A_hstep * 4]), bfloat16_to_float32(p0[A_hstep * 5]), bfloat16_to_float32(p0[A_hstep * 6]), bfloat16_to_float32(p0[A_hstep * 7]));
                    _p = _mm256_mul_ps(_p, _mm256_loadu_ps(ps));
                    const int64_t q = float2int8_avx(_mm256_mul_ps(_p, _scale256));
                    _mm_storel_epi64((__m128i*)pp, _mm_cvtsi64_si128(q));
                    pp += 8;
                    p0 += A_hstep * 8;
                    ps += 8;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
#if defined(__x86_64__) || defined(_M_X64)
                    __m128i _q8 = _mm_cvtsi64_si128(q);
#else
                    __m128i _q8 = _mm_loadl_epi64((const __m128i*)(pp - 8));
#endif
                    __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                    w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                }
#endif // __AVX2__
#endif // __AVX__
                __m128 _scale128 = _mm_set1_ps(scale);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p = _mm_setr_ps(bfloat16_to_float32(p0[0]), bfloat16_to_float32(p0[A_hstep]), bfloat16_to_float32(p0[A_hstep * 2]), bfloat16_to_float32(p0[A_hstep * 3]));
                    _p = _mm_mul_ps(_p, _mm_loadu_ps(ps));
                    const int32_t q = float2int8_sse(_mm_mul_ps(_p, _scale128));
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(q)));
                    pp += 4;
                    p0 += A_hstep * 4;
                    ps += 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                    __m128i _q8 = _mm_cvtsi32_si128(q);
                    __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                    w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
                }
#endif // __SSE2__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_mm_cvtsi32_si128(w_shift * 127)));
                    pp += 4;
                }
#endif
                for (; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(p0[0]);
                    v *= ps[0];
                    *pp++ = float2int8(v * scale);
                    p0 += A_hstep;
                    ps++;
                }

                pd++;
            }
        }
    }
}

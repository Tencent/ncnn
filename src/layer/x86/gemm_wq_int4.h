// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void pack_B_tile_wq_int4_avx512vnni(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void gemm_transB_packed_tile_wq_int4_avx512vnni(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVX512F__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_B_tile_wq_int4_avxvnniint8(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void gemm_transB_packed_tile_wq_int4_avxvnniint8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVX512F__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_B_tile_wq_int4_avxvnni(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void gemm_transB_packed_tile_wq_int4_avxvnni(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_B_tile_wq_int4_avx2(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void gemm_transB_packed_tile_wq_int4_avx2(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void gemm_transB_packed_tile_wq_int4_xop(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

static signed char get_int4_value_wq_int4(const unsigned char* p, int k)
{
    return (signed char)(p[k / 2] << ((k & 1) ? 0 : 4) & 0xf0);
}

#if __SSE2__
static __m128i decode_int4_wq_int4(__m128i _p)
{
    __m128i _v15 = _mm_set1_epi8(15);
    __m128i _p0 = _mm_slli_epi16(_mm_and_si128(_p, _v15), 4);
    __m128i _p1 = _mm_andnot_si128(_v15, _p);
    return _mm_unpacklo_epi8(_p0, _p1);
}

#if __AVX2__
static __m256i decode_int4x32_wq_int4(__m128i _p)
{
    __m256i _p16 = _mm256_cvtepu8_epi16(_p);
    __m256i _p0 = _mm256_slli_epi16(_p16, 4);
    __m256i _p1 = _mm256_slli_epi16(_p16, 8);
    return _mm256_and_si256(_mm256_or_si256(_p0, _p1), _mm256_set1_epi8(-16));
}
#endif // __AVX2__
#endif // __SSE2__

static void pack_B_tile_wq_int4(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_B_tile_wq_int4_avx512vnni(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVX512F__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        pack_B_tile_wq_int4_avxvnniint8(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVX512F__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_B_tile_wq_int4_avxvnni(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_B_tile_wq_int4_avx2(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

    const int block_count = (K + block_size - 1) / block_size;
    unsigned char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const float* ps = B_scales.row(j + jj);

        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(B.w));
        __m256i _sindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _sindex = _mm256_mullo_epi32(_sindex, _mm256_set1_epi32(B_scales.w));
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
        __m256i _v127 = _mm256_set1_epi8(127);
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
            __m256i _a_shift = _mm256_setzero_si256();
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256i _p = _mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(unsigned char));
                __m128i _packed0 = _mm256_comp_cvtepi32_epi16(_p);
                __m128i _packed1 = _mm256_comp_cvtepi32_epi16(_mm256_srli_epi32(_p, 16));
                _mm_storeu_si128((__m128i*)pp, _packed0);
                _mm_storeu_si128((__m128i*)(pp + 16), _packed1);
#if !__AVXVNNIINT8__
                _a_shift = _mm256_comp_dpbusd_epi32(_a_shift, _v127, decode_int4x32_wq_int4(_packed0));
                _a_shift = _mm256_comp_dpbusd_epi32(_a_shift, _v127, decode_int4x32_wq_int4(_packed1));
#endif // !__AVXVNNIINT8__
                pp += 32;
                p0 += 4;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _packed = _mm_setr_epi16(p0[0] | p0[1] << 8, p0[B.w] | p0[B.w + 1] << 8, p0[B.w * 2] | p0[B.w * 2 + 1] << 8, p0[B.w * 3] | p0[B.w * 3 + 1] << 8, p0[B.w * 4] | p0[B.w * 4 + 1] << 8, p0[B.w * 5] | p0[B.w * 5 + 1] << 8, p0[B.w * 6] | p0[B.w * 6 + 1] << 8, p0[B.w * 7] | p0[B.w * 7 + 1] << 8);
                _mm_storeu_si128((__m128i*)pp, _packed);
#if !__AVXVNNIINT8__
                _a_shift = _mm256_comp_dpbusd_epi32(_a_shift, _v127, decode_int4x32_wq_int4(_packed));
#endif // !__AVXVNNIINT8__
                pp += 16;
                p0 += 2;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm256_storeu_si256((__m256i*)pp, _a_shift);
                pp += 32;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _packed = _mm_setr_epi8(p0[0], p0[B.w], p0[B.w * 2], p0[B.w * 3], p0[B.w * 4], p0[B.w * 5], p0[B.w * 6], p0[B.w * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                _mm_storel_epi64((__m128i*)pp, _packed);
                pp += 8;
                p0++;
            }
            for (; kk < max_kk; kk++)
            {
                const unsigned int packed = (p0[0] & 15) | (p0[B.w] & 15) << 4 | (p0[B.w * 2] & 15) << 8 | (p0[B.w * 3] & 15) << 12 | (p0[B.w * 4] & 15) << 16 | (p0[B.w * 5] & 15) << 20 | (p0[B.w * 6] & 15) << 24 | (unsigned int)(p0[B.w * 7] & 15) << 28;
                __m128i _packed = _mm_cvtsi32_si128((int)packed);
                _mm_store_ss((float*)pp, _mm_castsi128_ps(_packed));
                pp += 4;
                p0++;
            }

            __m256 _scale = _mm256_i32gather_ps(ps, _sindex, sizeof(float));
            _mm256_storeu_ps(pd, _mm256_mul_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _scale), _mm256_set1_ps(0.0625f)));
            pd += 8;
            ps++;
        }
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);

#if __AVX2__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(B.w));
        __m128i _sindex = _mm_setr_epi32(0, 1, 2, 3);
        _sindex = _mm_mullo_epi32(_sindex, _mm_set1_epi32(B_scales.w));
#else
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);
#endif // __AVX2__
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
        __m128i _v127 = _mm_set1_epi8(127);
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
            __m128i _a_shift = _mm_setzero_si128();
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _p = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(unsigned char));
                __m128i _packed0 = _mm_comp_cvtepi32_epi16(_p);
                __m128i _packed1 = _mm_comp_cvtepi32_epi16(_mm_srli_epi32(_p, 16));
                _mm_storel_epi64((__m128i*)pp, _packed0);
                _mm_storel_epi64((__m128i*)(pp + 8), _packed1);
#if !__AVXVNNIINT8__
                _a_shift = _mm_comp_dpbusd_epi32(_a_shift, _v127, decode_int4_wq_int4(_packed0));
                _a_shift = _mm_comp_dpbusd_epi32(_a_shift, _v127, decode_int4_wq_int4(_packed1));
#endif // !__AVXVNNIINT8__
                pp += 16;
                p0 += 4;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVX2__
                __m128i _packed = _mm_setr_epi16(p0[0] | p0[1] << 8, p0[B.w] | p0[B.w + 1] << 8, p0[B.w * 2] | p0[B.w * 2 + 1] << 8, p0[B.w * 3] | p0[B.w * 3 + 1] << 8, 0, 0, 0, 0);
#else
                __m128i _packed = _mm_setr_epi16(p0[0] | p0[1] << 8, p1[0] | p1[1] << 8, p2[0] | p2[1] << 8, p3[0] | p3[1] << 8, 0, 0, 0, 0);
#endif // __AVX2__
                _mm_storel_epi64((__m128i*)pp, _packed);
#if !__AVXVNNIINT8__
                _a_shift = _mm_comp_dpbusd_epi32(_a_shift, _v127, decode_int4_wq_int4(_packed));
#endif // !__AVXVNNIINT8__
                pp += 8;
                p0 += 2;
#if !__AVX2__
                p1 += 2;
                p2 += 2;
                p3 += 2;
#endif // !__AVX2__
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm_storeu_si128((__m128i*)pp, _a_shift);
                pp += 16;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX2__
                const unsigned int packed = p0[0] | p0[B.w] << 8 | p0[B.w * 2] << 16 | (unsigned int)p0[B.w * 3] << 24;
#else
                const unsigned int packed = p0[0] | p1[0] << 8 | p2[0] << 16 | (unsigned int)p3[0] << 24;
#endif // __AVX2__
                __m128i _packed = _mm_cvtsi32_si128((int)packed);
                _mm_store_ss((float*)pp, _mm_castsi128_ps(_packed));
                pp += 4;
                p0++;
#if !__AVX2__
                p1++;
                p2++;
                p3++;
#endif // !__AVX2__
            }
            for (; kk < max_kk; kk++)
            {
#if __AVX2__
                pp[0] = (p0[0] & 15) | (p0[B.w] & 15) << 4;
                pp[1] = (p0[B.w * 2] & 15) | (p0[B.w * 3] & 15) << 4;
#else
                pp[0] = (p0[0] & 15) | (p1[0] & 15) << 4;
                pp[1] = (p2[0] & 15) | (p3[0] & 15) << 4;
#endif // __AVX2__
                pp += 2;
                p0++;
#if !__AVX2__
                p1++;
                p2++;
                p3++;
#endif // !__AVX2__
            }

#if __AVX2__
            __m128 _scale = _mm_i32gather_ps(ps0, _sindex, sizeof(float));
            _mm_storeu_ps(pd, _mm_mul_ps(_mm_div_ps(_mm_set1_ps(1.f), _scale), _mm_set1_ps(0.0625f)));
            ps0++;
#else
            pd[0] = (1.f / *ps0++) * 0.0625f;
            pd[1] = (1.f / *ps1++) * 0.0625f;
            pd[2] = (1.f / *ps2++) * 0.0625f;
            pd[3] = (1.f / *ps3++) * 0.0625f;
#endif // __AVX2__
            pd += 4;
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
            __m128i _v127 = _mm_set1_epi8(127);
            __m128i _a_shift = _mm_setzero_si128();
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
#endif // __SSE2__
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                const unsigned int packed = p0[0] | p0[1] << 8 | p1[0] << 16 | (unsigned int)p1[1] << 24;
                __m128i _packed = _mm_cvtsi32_si128((int)packed);
                _mm_store_ss((float*)pp, _mm_castsi128_ps(_packed));
#if !__AVXVNNIINT8__
                _a_shift = _mm_comp_dpbusd_epi32(_a_shift, _v127, decode_int4_wq_int4(_packed));
#endif // !__AVXVNNIINT8__
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm_storel_epi64((__m128i*)pp, _a_shift);
                pp += 8;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp += 2;
            }
#endif // __SSE2__
#if __SSE2__
            for (; kk < max_kk; kk++)
            {
                *pp++ = (p0[0] & 15) | (p1[0] & 15) << 4;
                p0++;
                p1++;
            }
#else
            for (; kk < max_kk; kk++)
            {
                const int shift = (kk & 1) * 4;
                *pp++ = ((p0[0] >> shift) & 15) | ((p1[0] >> shift) & 15) << 4;
                if (kk & 1)
                {
                    p0++;
                    p1++;
                }
            }
            if (max_kk & 1)
            {
                p0++;
                p1++;
            }
#endif // __SSE2__

            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
            __m128i _v127 = _mm_set1_epi8(127);
            __m128i _a_shift = _mm_setzero_si128();
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
#endif // __SSE2__
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
#if !__AVXVNNIINT8__
                __m128i _packed = _mm_cvtsi32_si128(p0[0] | p0[1] << 8);
                _a_shift = _mm_comp_dpbusd_epi32(_a_shift, _v127, decode_int4_wq_int4(_packed));
#endif // !__AVXVNNIINT8__
                pp += 2;
                p0 += 2;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm_store_ss((float*)pp, _mm_castsi128_ps(_a_shift));
                pp += 4;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
#endif // __SSE2__
            for (; kk + 1 < max_kk; kk += 2)
            {
                *pp = *p0++;
                pp++;
            }
            for (; kk < max_kk; kk++)
                *pp++ = *p0++ & 15;

            *pd++ = (1.f / *ps0++) * 0.0625f;
        }
    }
}

static void gemm_transB_packed_tile_wq_int4(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        gemm_transB_packed_tile_wq_int4_avx512vnni(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVX512F__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        gemm_transB_packed_tile_wq_int4_avxvnniint8(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVX512F__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        gemm_transB_packed_tile_wq_int4_avxvnni(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        gemm_transB_packed_tile_wq_int4_avx2(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_xop())
    {
        gemm_transB_packed_tile_wq_int4_xop(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

    const signed char* pAT = AT_tile;
    const int A_hstep = AT_tile.w;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = AT_descales_tile.w;
    const unsigned char* pBT = BT_tile;
    int correction_block_count = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
    correction_block_count = (K + block_size - 4) / block_size;
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int block_count = (K + block_size - 1) / block_size;
    const int block_start = k / block_size;
    int b_offset = k / 2;
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
    b_offset += block_start * 4;
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 8;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 8;
            __m512 _fsum0;
            __m512 _fsum1;
            __m512 _fsum2;
            __m512 _fsum3;
            __m512 _fsum4;
            __m512 _fsum5;
            __m512 _fsum6;
            __m512 _fsum7;

            if (k == 0)
            {
                _fsum0 = _mm512_setzero_ps();
                _fsum1 = _mm512_setzero_ps();
                _fsum2 = _mm512_setzero_ps();
                _fsum3 = _mm512_setzero_ps();
                _fsum4 = _mm512_setzero_ps();
                _fsum5 = _mm512_setzero_ps();
                _fsum6 = _mm512_setzero_ps();
                _fsum7 = _mm512_setzero_ps();
            }
            else
            {
                _fsum0 = _mm512_loadu_ps(outptr);
                _fsum1 = _mm512_loadu_ps(outptr + 16);
                _fsum2 = _mm512_loadu_ps(outptr + 32);
                _fsum3 = _mm512_loadu_ps(outptr + 48);
                _fsum4 = _mm512_loadu_ps(outptr + 64);
                _fsum5 = _mm512_loadu_ps(outptr + 80);
                _fsum6 = _mm512_loadu_ps(outptr + 96);
                _fsum7 = _mm512_loadu_ps(outptr + 112);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m512i _sum0 = _mm512_setzero_si512();
                __m512i _sum1 = _mm512_setzero_si512();
                __m512i _sum2 = _mm512_setzero_si512();
                __m512i _sum3 = _mm512_setzero_si512();
                __m512i _sum4 = _mm512_setzero_si512();
                __m512i _sum5 = _mm512_setzero_si512();
                __m512i _sum6 = _mm512_setzero_si512();
                __m512i _sum7 = _mm512_setzero_si512();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;

                // from
                //      00 01 02 03 04 05 06 07
                //      10 11 12 13 14 15 16 17
                //      ...
                //      f0 f1 f2 f3 f4 f5 f6 f7
                //
                // to
                // _sum0 00 11 22 33  44 55 66 77  80 91 a2 b3  c4 d5 e6 f7
                // _sum1 01 12 23 30  45 56 67 74  81 92 a3 b0  c5 d6 e7 f4
                // _sum2 20 31 02 13  64 75 46 57  a0 b1 82 93  e4 f5 c6 d7
                // _sum3 21 32 03 10  65 76 47 54  a1 b2 83 90  e5 f6 c7 d4
                // _sum4 04 15 26 37  40 51 62 73  84 95 a6 b7  c0 d1 e2 f3
                // _sum5 05 16 27 34  41 52 63 70  85 96 a7 b4  c1 d2 e3 f0
                // _sum6 24 35 06 17  60 71 42 53  a4 b5 86 97  e0 f1 c2 d3
                // _sum7 25 36 07 14  61 72 43 50  a5 b6 87 94  e1 f2 c3 d0
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m256i _pB = decode_int4x32_wq_int4(_mm_loadu_si128((const __m128i*)pB));
                    __m512i _pB0 = combine8x2_epi32(_pB, _pB);
                    __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _pB3 = _mm512_alignr_epi8(_pB2, _pB2, 4);
                    _sum0 = _mm512_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_dpbusd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_dpbusd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm512_dpbusd_epi32(_sum3, _pA1, _pB1);
                    _sum4 = _mm512_dpbusd_epi32(_sum4, _pA0, _pB2);
                    _sum5 = _mm512_dpbusd_epi32(_sum5, _pA0, _pB3);
                    _sum6 = _mm512_dpbusd_epi32(_sum6, _pA1, _pB2);
                    _sum7 = _mm512_dpbusd_epi32(_sum7, _pA1, _pB3);
                    pB += 16;
                    pA += 64;
                }
                if (max_kk0 >= 4)
                {
                    __m256i _a_shift = _mm256_loadu_si256((const __m256i*)pB);
                    __m512i _a_shift0 = combine8x2_epi32(_a_shift, _a_shift);
                    __m512i _a_shift1 = _mm512_alignr_epi8(_a_shift0, _a_shift0, 4);
                    __m512i _a_shift2 = _mm512_permutex_epi64(_a_shift0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _a_shift3 = _mm512_alignr_epi8(_a_shift2, _a_shift2, 4);
                    _sum0 = _mm512_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm512_sub_epi32(_sum1, _a_shift1);
                    _sum2 = _mm512_sub_epi32(_sum2, _a_shift0);
                    _sum3 = _mm512_sub_epi32(_sum3, _a_shift1);
                    _sum4 = _mm512_sub_epi32(_sum4, _a_shift2);
                    _sum5 = _mm512_sub_epi32(_sum5, _a_shift3);
                    _sum6 = _mm512_sub_epi32(_sum6, _a_shift2);
                    _sum7 = _mm512_sub_epi32(_sum7, _a_shift3);
                    pB += 32;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                    __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                    __m128i _pB = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
                    __m256i _pBB = _mm256_cvtepi8_epi16(_pB);
                    __m512i _pB0 = combine8x2_epi32(_pBB, _pBB);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _pB3 = _mm512_alignr_epi8(_pB2, _pB2, 4);
                    _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm512_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    _sum4 = _mm512_comp_dpwssd_epi32(_sum4, _pA0, _pB2);
                    _sum5 = _mm512_comp_dpwssd_epi32(_sum5, _pA0, _pB3);
                    _sum6 = _mm512_comp_dpwssd_epi32(_sum6, _pA1, _pB2);
                    _sum7 = _mm512_comp_dpwssd_epi32(_sum7, _pA1, _pB3);
                    pB += 8;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m128i _pB = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    _pB = _mm_cvtepi8_epi16(_pB);
                    __m256i _pB0 = combine4x2_epi32(_pB, _pB);
                    __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB2 = _mm256_alignr_epi8(_pB0, _pB0, 8);
                    __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0)));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1)));
                    _sum4 = _mm512_add_epi32(_sum4, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2)));
                    _sum5 = _mm512_add_epi32(_sum5, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3)));
                    _sum6 = _mm512_add_epi32(_sum6, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB2)));
                    _sum7 = _mm512_add_epi32(_sum7, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB3)));
                    pB += 4;
                    pA += 16;
                }

                __m512 _descaleA0 = _mm512_loadu_ps(pA_descales);
                __m512 _descaleA1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_descaleA0), _mm512_castps_si512(_descaleA0), 8));
                __m256 _descaleB8 = _mm256_loadu_ps(pB_descales);
                __m512 _descaleB0 = combine8x2_ps(_descaleB8, _descaleB8);
                __m512 _descaleB1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_descaleB0), _mm512_castps_si512(_descaleB0), 4));
                __m512 _descaleB2 = _mm512_castsi512_ps(_mm512_permutex_epi64(_mm512_castps_si512(_descaleB0), _MM_SHUFFLE(1, 0, 3, 2)));
                __m512 _descaleB3 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_descaleB2), _mm512_castps_si512(_descaleB2), 4));
                _fsum0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum0), _mm512_mul_ps(_descaleA0, _descaleB0), _fsum0);
                _fsum1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum1), _mm512_mul_ps(_descaleA0, _descaleB1), _fsum1);
                _fsum2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum2), _mm512_mul_ps(_descaleA1, _descaleB0), _fsum2);
                _fsum3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum3), _mm512_mul_ps(_descaleA1, _descaleB1), _fsum3);
                _fsum4 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum4), _mm512_mul_ps(_descaleA0, _descaleB2), _fsum4);
                _fsum5 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum5), _mm512_mul_ps(_descaleA0, _descaleB3), _fsum5);
                _fsum6 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum6), _mm512_mul_ps(_descaleA1, _descaleB2), _fsum6);
                _fsum7 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum7), _mm512_mul_ps(_descaleA1, _descaleB3), _fsum7);
                pA_descales += 16;
                pB_descales += 8;
            }

            _mm512_storeu_ps(outptr + 0, _fsum0);
            _mm512_storeu_ps(outptr + 16, _fsum1);
            _mm512_storeu_ps(outptr + 32, _fsum2);
            _mm512_storeu_ps(outptr + 48, _fsum3);
            _mm512_storeu_ps(outptr + 64, _fsum4);
            _mm512_storeu_ps(outptr + 80, _fsum5);
            _mm512_storeu_ps(outptr + 96, _fsum6);
            _mm512_storeu_ps(outptr + 112, _fsum7);
            outptr += 128;
            pB_panel += ((size_t)8 * K + 1) / 2 + (size_t)8 * 4 * correction_block_count;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 4;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 4;
            __m512 _fsum0;
            __m512 _fsum1;
            __m512 _fsum2;
            __m512 _fsum3;

            if (k == 0)
            {
                _fsum0 = _mm512_setzero_ps();
                _fsum1 = _mm512_setzero_ps();
                _fsum2 = _mm512_setzero_ps();
                _fsum3 = _mm512_setzero_ps();
            }
            else
            {
                _fsum0 = _mm512_loadu_ps(outptr);
                _fsum1 = _mm512_loadu_ps(outptr + 16);
                _fsum2 = _mm512_loadu_ps(outptr + 32);
                _fsum3 = _mm512_loadu_ps(outptr + 48);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m512i _sum0 = _mm512_setzero_si512();
                __m512i _sum1 = _mm512_setzero_si512();
                __m512i _sum2 = _mm512_setzero_si512();
                __m512i _sum3 = _mm512_setzero_si512();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB0 = _mm512_broadcast_i32x4(decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB)));
                    __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm512_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_dpbusd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_dpbusd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm512_dpbusd_epi32(_sum3, _pA1, _pB1);
                    pB += 8;
                    pA += 64;
                }
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift = _mm_loadu_si128((const __m128i*)pB);
                    __m512i _a_shift0 = _mm512_broadcast_i32x4(_a_shift);
                    __m512i _a_shift1 = _mm512_alignr_epi8(_a_shift0, _a_shift0, 4);
                    _sum0 = _mm512_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm512_sub_epi32(_sum1, _a_shift1);
                    _sum2 = _mm512_sub_epi32(_sum2, _a_shift0);
                    _sum3 = _mm512_sub_epi32(_sum3, _a_shift1);
                    pB += 16;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                    __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    _pB8 = _mm_unpacklo_epi64(_pB8, _pB8);
                    __m256i _pB = combine4x2_epi32(_pB8, _pB8);
                    __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm512_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pB += 4;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8)), _MM_SHUFFLE(0, 0, 0, 0)));
                    __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0)));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1)));
                    pB += 2;
                    pA += 16;
                }

                __m512 _descaleA0 = _mm512_loadu_ps(pA_descales);
                __m512 _descaleA1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_descaleA0), _mm512_castps_si512(_descaleA0), 8));
                __m512 _descaleB0 = _mm512_broadcast_f32x4(_mm_loadu_ps(pB_descales));
                __m512 _descaleB1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_descaleB0), _mm512_castps_si512(_descaleB0), 4));
                _fsum0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum0), _mm512_mul_ps(_descaleA0, _descaleB0), _fsum0);
                _fsum1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum1), _mm512_mul_ps(_descaleA0, _descaleB1), _fsum1);
                _fsum2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum2), _mm512_mul_ps(_descaleA1, _descaleB0), _fsum2);
                _fsum3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum3), _mm512_mul_ps(_descaleA1, _descaleB1), _fsum3);
                pA_descales += 16;
                pB_descales += 4;
            }

            _mm512_storeu_ps(outptr + 0, _fsum0);
            _mm512_storeu_ps(outptr + 16, _fsum1);
            _mm512_storeu_ps(outptr + 32, _fsum2);
            _mm512_storeu_ps(outptr + 48, _fsum3);
            outptr += 64;
            pB_panel += ((size_t)4 * K + 1) / 2 + (size_t)4 * 4 * correction_block_count;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 2;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 2;
            __m512 _fsum0;
            __m512 _fsum1;

            if (k == 0)
            {
                _fsum0 = _mm512_setzero_ps();
                _fsum1 = _mm512_setzero_ps();
            }
            else
            {
                _fsum0 = _mm512_loadu_ps(outptr);
                _fsum1 = _mm512_loadu_ps(outptr + 16);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m512i _sum0 = _mm512_setzero_si512();
                __m512i _sum1 = _mm512_setzero_si512();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB0 = _mm512_broadcastq_epi64(decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB))));
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm512_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_dpbusd_epi32(_sum1, _pA0, _pB1);
                    pB += 4;
                    pA += 64;
                }
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift = _mm_loadl_epi64((const __m128i*)pB);
                    _a_shift = _mm_shuffle_epi32(_a_shift, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _a_shift0 = _mm512_broadcast_i32x4(_a_shift);
                    __m512i _a_shift1 = _mm512_alignr_epi8(_a_shift0, _a_shift0, 4);
                    _sum0 = _mm512_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm512_sub_epi32(_sum1, _a_shift1);
                    pB += 8;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                    __m256i _pB = _mm256_set1_epi32(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8))));
                    __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    pB += 2;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m128i _pB = _mm_set1_epi16(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]))) & 0xffff);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);
                    __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                    pB += 1;
                    pA += 16;
                }

                __m512 _descaleA0 = _mm512_loadu_ps(pA_descales);
                __m512 _descaleB0 = _mm512_castsi512_ps(_mm512_broadcastq_epi64(_mm_loadl_epi64((const __m128i*)pB_descales)));
                __m512 _descaleB1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_descaleB0), _mm512_castps_si512(_descaleB0), 4));
                _fsum0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum0), _mm512_mul_ps(_descaleA0, _descaleB0), _fsum0);
                _fsum1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum1), _mm512_mul_ps(_descaleA0, _descaleB1), _fsum1);
                pA_descales += 16;
                pB_descales += 2;
            }

            _mm512_storeu_ps(outptr + 0, _fsum0);
            _mm512_storeu_ps(outptr + 16, _fsum1);
            outptr += 32;
            pB_panel += ((size_t)2 * K + 1) / 2 + (size_t)2 * 4 * correction_block_count;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset;
            const float* pB_descales = pB_descales_panel + (size_t)block_start;
            __m512 _fsum0;

            if (k == 0)
            {
                _fsum0 = _mm512_setzero_ps();
            }
            else
            {
                _fsum0 = _mm512_loadu_ps(outptr);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m512i _sum0 = _mm512_setzero_si512();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB0 = _mm512_set1_epi32(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8))));
                    _sum0 = _mm512_dpbusd_epi32(_sum0, _pA0, _pB0);
                    pB += 2;
                    pA += 64;
                }
                if (max_kk0 >= 4)
                {
                    _sum0 = _mm512_sub_epi32(_sum0, _mm512_set1_epi32(_mm_cvtsi128_si32(_mm_castps_si128(_mm_load_ss((const float*)pB)))));
                    pB += 4;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                    __m512i _pB0 = _mm512_cvtepi8_epi16(_mm256_broadcastsi128_si256(_mm_set1_epi16(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]))) & 0xffff)));
                    _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    pB += 1;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m512i _pA0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)pA));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_mullo_epi32(_pA0, _mm512_set1_epi32(get_int4_value_wq_int4(pB, 0))));
                    pB += 1;
                    pA += 16;
                }

                __m512 _descaleA0 = _mm512_loadu_ps(pA_descales);
                __m512 _descaleB0 = _mm512_set1_ps(pB_descales[0]);
                _fsum0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_sum0), _mm512_mul_ps(_descaleA0, _descaleB0), _fsum0);
                pA_descales += 16;
                pB_descales += 1;
            }

            _mm512_storeu_ps(outptr + 0, _fsum0);
            outptr += 16;
            pB_panel += ((size_t)K + 1) / 2 + (size_t)4 * correction_block_count;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 16;
        pAT_descales += A_descales_hstep * 16;
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 8;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 8;
            __m256 _fsum0;
            __m256 _fsum1;
            __m256 _fsum2;
            __m256 _fsum3;
            __m256 _fsum4;
            __m256 _fsum5;
            __m256 _fsum6;
            __m256 _fsum7;

            if (k == 0)
            {
                _fsum0 = _mm256_setzero_ps();
                _fsum1 = _mm256_setzero_ps();
                _fsum2 = _mm256_setzero_ps();
                _fsum3 = _mm256_setzero_ps();
                _fsum4 = _mm256_setzero_ps();
                _fsum5 = _mm256_setzero_ps();
                _fsum6 = _mm256_setzero_ps();
                _fsum7 = _mm256_setzero_ps();
            }
            else
            {
                _fsum0 = _mm256_loadu_ps(outptr);
                _fsum1 = _mm256_loadu_ps(outptr + 8);
                _fsum2 = _mm256_loadu_ps(outptr + 16);
                _fsum3 = _mm256_loadu_ps(outptr + 24);
                _fsum4 = _mm256_loadu_ps(outptr + 32);
                _fsum5 = _mm256_loadu_ps(outptr + 40);
                _fsum6 = _mm256_loadu_ps(outptr + 48);
                _fsum7 = _mm256_loadu_ps(outptr + 56);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                __m256i _sum2 = _mm256_setzero_si256();
                __m256i _sum3 = _mm256_setzero_si256();
                __m256i _sum4 = _mm256_setzero_si256();
                __m256i _sum5 = _mm256_setzero_si256();
                __m256i _sum6 = _mm256_setzero_si256();
                __m256i _sum7 = _mm256_setzero_si256();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB0 = decode_int4x32_wq_int4(_mm_loadu_si128((const __m128i*)pB));
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB3 = _mm256_alignr_epi8(_pB2, _pB2, 4);
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _pA1, _pB1);
                    _sum4 = _mm256_comp_dpbusd_epi32(_sum4, _pA0, _pB2);
                    _sum5 = _mm256_comp_dpbusd_epi32(_sum5, _pA0, _pB3);
                    _sum6 = _mm256_comp_dpbusd_epi32(_sum6, _pA1, _pB2);
                    _sum7 = _mm256_comp_dpbusd_epi32(_sum7, _pA1, _pB3);
                    pB += 16;
                    pA += 32;
                }
                if (max_kk0 >= 4)
                {
                    __m256i _a_shift0 = _mm256_loadu_si256((const __m256i*)pB);
                    __m256i _a_shift1 = _mm256_alignr_epi8(_a_shift0, _a_shift0, 4);
                    __m256i _a_shift2 = _mm256_permute4x64_epi64(_a_shift0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _a_shift3 = _mm256_alignr_epi8(_a_shift2, _a_shift2, 4);
                    _sum0 = _mm256_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _a_shift1);
                    _sum2 = _mm256_sub_epi32(_sum2, _a_shift0);
                    _sum3 = _mm256_sub_epi32(_sum3, _a_shift1);
                    _sum4 = _mm256_sub_epi32(_sum4, _a_shift2);
                    _sum5 = _mm256_sub_epi32(_sum5, _a_shift3);
                    _sum6 = _mm256_sub_epi32(_sum6, _a_shift2);
                    _sum7 = _mm256_sub_epi32(_sum7, _a_shift3);
                    pB += 32;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA8);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB8);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB3 = _mm256_alignr_epi8(_pB2, _pB2, 4);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    _sum4 = _mm256_comp_dpwssd_epi32(_sum4, _pA0, _pB2);
                    _sum5 = _mm256_comp_dpwssd_epi32(_sum5, _pA0, _pB3);
                    _sum6 = _mm256_comp_dpwssd_epi32(_sum6, _pA1, _pB2);
                    _sum7 = _mm256_comp_dpwssd_epi32(_sum7, _pA1, _pB3);
                    pB += 8;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA0 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB0 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    _pA0 = _mm_cvtepi8_epi16(_pA0);
                    _pB0 = _mm_cvtepi8_epi16(_pB0);
                    __m128i _pA1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pA0, _MM_SHUFFLE(1, 0, 3, 2)), _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    __m128i _pB2 = _mm_alignr_epi8(_pB0, _pB0, 8);
                    __m128i _pB3 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0)));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1)));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0)));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1)));
                    _sum4 = _mm256_add_epi32(_sum4, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB2)));
                    _sum5 = _mm256_add_epi32(_sum5, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB3)));
                    _sum6 = _mm256_add_epi32(_sum6, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB2)));
                    _sum7 = _mm256_add_epi32(_sum7, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB3)));
                    pB += 4;
                    pA += 8;
                }

                __m256 _descaleA0 = _mm256_loadu_ps(pA_descales);
                __m256 _descaleA1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_descaleA0), _mm256_castps_si256(_descaleA0), 8));
                __m256 _descaleB0 = _mm256_loadu_ps(pB_descales);
                __m256 _descaleB1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_descaleB0), _mm256_castps_si256(_descaleB0), 4));
                __m256 _descaleB2 = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(_descaleB0), _MM_SHUFFLE(1, 0, 3, 2)));
                __m256 _descaleB3 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_descaleB2), _mm256_castps_si256(_descaleB2), 4));
                _fsum0 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_descaleA0, _descaleB0), _fsum0);
                _fsum1 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_descaleA0, _descaleB1), _fsum1);
                _fsum2 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum2), _mm256_mul_ps(_descaleA1, _descaleB0), _fsum2);
                _fsum3 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum3), _mm256_mul_ps(_descaleA1, _descaleB1), _fsum3);
                _fsum4 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum4), _mm256_mul_ps(_descaleA0, _descaleB2), _fsum4);
                _fsum5 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum5), _mm256_mul_ps(_descaleA0, _descaleB3), _fsum5);
                _fsum6 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum6), _mm256_mul_ps(_descaleA1, _descaleB2), _fsum6);
                _fsum7 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum7), _mm256_mul_ps(_descaleA1, _descaleB3), _fsum7);
                pA_descales += 8;
                pB_descales += 8;
            }

            _mm256_storeu_ps(outptr + 0, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            _mm256_storeu_ps(outptr + 16, _fsum2);
            _mm256_storeu_ps(outptr + 24, _fsum3);
            _mm256_storeu_ps(outptr + 32, _fsum4);
            _mm256_storeu_ps(outptr + 40, _fsum5);
            _mm256_storeu_ps(outptr + 48, _fsum6);
            _mm256_storeu_ps(outptr + 56, _fsum7);
            outptr += 64;
            pB_panel += ((size_t)8 * K + 1) / 2 + (size_t)8 * 4 * correction_block_count;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __AVX512F__
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 4;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 4;
            __m256 _fsum0;
            __m256 _fsum1;
            __m256 _fsum2;
            __m256 _fsum3;

            if (k == 0)
            {
                _fsum0 = _mm256_setzero_ps();
                _fsum1 = _mm256_setzero_ps();
                _fsum2 = _mm256_setzero_ps();
                _fsum3 = _mm256_setzero_ps();
            }
            else
            {
                _fsum0 = _mm256_loadu_ps(outptr);
                _fsum1 = _mm256_loadu_ps(outptr + 8);
                _fsum2 = _mm256_loadu_ps(outptr + 16);
                _fsum3 = _mm256_loadu_ps(outptr + 24);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                __m256i _sum2 = _mm256_setzero_si256();
                __m256i _sum3 = _mm256_setzero_si256();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m128i _pB = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
                    __m256i _pB0 = combine4x2_epi32(_pB, _pB);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
#if __AVXVNNIINT8__
                    _sum0 = _mm256_dpbssd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm256_dpbssd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm256_dpbssd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm256_dpbssd_epi32(_sum3, _pB1, _pA1);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _pA1, _pB1);
#endif // __AVXVNNIINT8__
                    pB += 8;
                    pA += 32;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift = _mm_loadu_si128((const __m128i*)pB);
                    __m256i _a_shift0 = combine4x2_epi32(_a_shift, _a_shift);
                    __m256i _a_shift1 = _mm256_alignr_epi8(_a_shift0, _a_shift0, 4);
                    _sum0 = _mm256_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _a_shift1);
                    _sum2 = _mm256_sub_epi32(_sum2, _a_shift0);
                    _sum3 = _mm256_sub_epi32(_sum3, _a_shift1);
                    pB += 16;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = _mm_shuffle_epi32(decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB))), _MM_SHUFFLE(1, 0, 1, 0));
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pB += 4;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA0 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB0 = _mm_shuffle_epi32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8)), _MM_SHUFFLE(0, 0, 0, 0));
                    _pA0 = _mm_cvtepi8_epi16(_pA0);
                    _pB0 = _mm_cvtepi8_epi16(_pB0);
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0)));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1)));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0)));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1)));
                    pB += 2;
                    pA += 8;
                }

                __m256 _descaleA0 = _mm256_loadu_ps(pA_descales);
                __m256 _descaleA1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_descaleA0), _mm256_castps_si256(_descaleA0), 8));
                __m128 _descaleB4 = _mm_loadu_ps(pB_descales);
                __m256 _descaleB0 = combine4x2_ps(_descaleB4, _descaleB4);
                __m256 _descaleB1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_descaleB0), _mm256_castps_si256(_descaleB0), 4));
                _fsum0 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_descaleA0, _descaleB0), _fsum0);
                _fsum1 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_descaleA0, _descaleB1), _fsum1);
                _fsum2 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum2), _mm256_mul_ps(_descaleA1, _descaleB0), _fsum2);
                _fsum3 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum3), _mm256_mul_ps(_descaleA1, _descaleB1), _fsum3);
                pA_descales += 8;
                pB_descales += 4;
            }

            _mm256_storeu_ps(outptr + 0, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            _mm256_storeu_ps(outptr + 16, _fsum2);
            _mm256_storeu_ps(outptr + 24, _fsum3);
            outptr += 32;
            pB_panel += ((size_t)4 * K + 1) / 2 + (size_t)4 * 4 * correction_block_count;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 2;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 2;
            __m256 _fsum0;
            __m256 _fsum1;

            if (k == 0)
            {
                _fsum0 = _mm256_setzero_ps();
                _fsum1 = _mm256_setzero_ps();
            }
            else
            {
                _fsum0 = _mm256_loadu_ps(outptr);
                _fsum1 = _mm256_loadu_ps(outptr + 8);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    _pB8 = _mm_unpacklo_epi64(_pB8, _pB8);
                    __m256i _pB0 = combine4x2_epi32(_pB8, _pB8);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
#if __AVXVNNIINT8__
                    _sum0 = _mm256_dpbssd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm256_dpbssd_epi32(_sum1, _pB1, _pA0);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pA0, _pB1);
#endif // __AVXVNNIINT8__
                    pB += 4;
                    pA += 32;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift = _mm_loadl_epi64((const __m128i*)pB);
                    _a_shift = _mm_shuffle_epi32(_a_shift, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256i _a_shift0 = _mm256_broadcastsi128_si256(_a_shift);
                    __m256i _a_shift1 = _mm256_alignr_epi8(_a_shift0, _a_shift0, 4);
                    _sum0 = _mm256_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _a_shift1);
                    pB += 8;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = _mm_shuffle_epi32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8)), _MM_SHUFFLE(0, 0, 0, 0));
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    pB += 2;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB0 = _mm_set1_epi16(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]))) & 0xffff);
                    _pA = _mm_cvtepi8_epi16(_pA);
                    _pB0 = _mm_cvtepi8_epi16(_pB0);
                    __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0)));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1)));
                    pB += 1;
                    pA += 8;
                }

                __m256 _descaleA0 = _mm256_loadu_ps(pA_descales);
                __m256 _descaleB0 = _mm256_castpd_ps(_mm256_broadcast_sd((const double*)pB_descales));
                __m256 _descaleB1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_descaleB0), _mm256_castps_si256(_descaleB0), 4));
                _fsum0 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_descaleA0, _descaleB0), _fsum0);
                _fsum1 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_descaleA0, _descaleB1), _fsum1);
                pA_descales += 8;
                pB_descales += 2;
            }

            _mm256_storeu_ps(outptr + 0, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            outptr += 16;
            pB_panel += ((size_t)2 * K + 1) / 2 + (size_t)2 * 4 * correction_block_count;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset;
            const float* pB_descales = pB_descales_panel + (size_t)block_start;
            __m256 _fsum0;

            if (k == 0)
            {
                _fsum0 = _mm256_setzero_ps();
            }
            else
            {
                _fsum0 = _mm256_loadu_ps(outptr);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pB = _mm256_set1_epi32(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8))));
#if __AVXVNNIINT8__
                    _sum0 = _mm256_dpbssd_epi32(_sum0, _pB, _pA);
#else // __AVXVNNIINT8__
#if __AVX512VNNI__ && _MSC_VER < 1932
                    // old msvc crash here  --- nihui
                    __m512i _pA0 = _mm512_cvtepu8_epi16(_pA);
                    __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);
                    __m512i _s0 = _mm512_madd_epi16(_pA0, _pB0);
                    __m256i _s1 = _mm256_hadd_epi32(_mm512_extracti32x8_epi32(_s0, 0), _mm512_extracti32x8_epi32(_s0, 1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_permute4x64_epi64(_s1, _MM_SHUFFLE(3, 1, 2, 0)));
#else
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pA, _pB);
#endif
#endif // __AVXVNNIINT8__
                    pB += 2;
                    pA += 32;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    _sum0 = _mm256_sub_epi32(_sum0, _mm256_set1_epi32(_mm_cvtsi128_si32(_mm_castps_si128(_mm_load_ss((const float*)pB)))));
                    pB += 4;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_mm_set1_epi16(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]))) & 0xffff));
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    pB += 1;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                    _pA = _mm_cvtepi8_epi16(_pA);
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _mm_set1_epi16(get_int4_value_wq_int4(pB, 0)))));
                    pB += 1;
                    pA += 8;
                }

                __m256 _descaleA0 = _mm256_loadu_ps(pA_descales);
                _fsum0 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_descaleA0, _mm256_set1_ps(pB_descales[0])), _fsum0);
                pA_descales += 8;
                pB_descales++;
            }

            _mm256_storeu_ps(outptr, _fsum0);
            outptr += 8;
            pB_panel += ((size_t)K + 1) / 2 + (size_t)4 * correction_block_count;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 8;
        pAT_descales += A_descales_hstep * 8;
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 8;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 8;
            __m256 _fsum0;
            __m256 _fsum1;
            __m256 _fsum2;
            __m256 _fsum3;

            if (k == 0)
            {
                _fsum0 = _mm256_setzero_ps();
                _fsum1 = _mm256_setzero_ps();
                _fsum2 = _mm256_setzero_ps();
                _fsum3 = _mm256_setzero_ps();
            }
            else
            {
                _fsum0 = _mm256_loadu_ps(outptr);
                _fsum1 = _mm256_loadu_ps(outptr + 8);
                _fsum2 = _mm256_loadu_ps(outptr + 16);
                _fsum3 = _mm256_loadu_ps(outptr + 24);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                __m256i _sum2 = _mm256_setzero_si256();
                __m256i _sum3 = _mm256_setzero_si256();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)pA));
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB0 = decode_int4x32_wq_int4(_mm_loadu_si128((const __m128i*)pB));
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _pA1, _pB1);
                    pA += 16;
                    pB += 16;
                }
                if (max_kk0 >= 4)
                {
                    __m256i _a_shift0 = _mm256_loadu_si256((const __m256i*)pB);
                    __m256i _a_shift1 = _mm256_alignr_epi8(_a_shift0, _a_shift0, 4);
                    _sum0 = _mm256_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _a_shift1);
                    _sum2 = _mm256_sub_epi32(_sum2, _a_shift0);
                    _sum3 = _mm256_sub_epi32(_sum3, _a_shift1);
                    pB += 32;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8x1 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pA8 = _mm_unpacklo_epi64(_pA8x1, _pA8x1);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA8);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB8);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pB8 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    __m128i _pA32 = _mm_cvtepi8_epi32(_pA8);
                    __m256i _pA0 = combine4x2_epi32(_pA32, _pA32);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB0 = combine4x2_epi32(_mm_cvtepi8_epi32(_pB8), _mm_cvtepi8_epi32(_mm_srli_si128(_pB8, 4)));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_mullo_epi32(_pA0, _pB0));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_mullo_epi32(_pA0, _pB1));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_mullo_epi32(_pA1, _pB0));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_mullo_epi32(_pA1, _pB1));
                    pA += 4;
                    pB += 4;
                }

                __m128 _descaleA128 = _mm_loadu_ps(pA_descales);
                __m256 _descaleA0 = combine4x2_ps(_descaleA128, _descaleA128);
                __m256 _descaleA1 = _mm256_shuffle_ps(_descaleA0, _descaleA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _descaleB0 = _mm256_loadu_ps(pB_descales);
                __m256 _descaleB1 = _mm256_shuffle_ps(_descaleB0, _descaleB0, _MM_SHUFFLE(0, 3, 2, 1));
                _fsum0 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_descaleA0, _descaleB0), _fsum0);
                _fsum1 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_descaleA0, _descaleB1), _fsum1);
                _fsum2 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum2), _mm256_mul_ps(_descaleA1, _descaleB0), _fsum2);
                _fsum3 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum3), _mm256_mul_ps(_descaleA1, _descaleB1), _fsum3);
                pA_descales += 4;
                pB_descales += 8;
            }

            _mm256_storeu_ps(outptr, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            _mm256_storeu_ps(outptr + 16, _fsum2);
            _mm256_storeu_ps(outptr + 24, _fsum3);
            outptr += 32;
            pB_panel += ((size_t)8 * K + 1) / 2 + (size_t)8 * 4 * correction_block_count;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __AVX512F__
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 4;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 4;
            __m128 _fsum0;
            __m128 _fsum1;
            __m128 _fsum2;
            __m128 _fsum3;

            if (k == 0)
            {
                _fsum0 = _mm_setzero_ps();
                _fsum1 = _mm_setzero_ps();
                _fsum2 = _mm_setzero_ps();
                _fsum3 = _mm_setzero_ps();
            }
            else
            {
                _fsum0 = _mm_loadu_ps(outptr);
                _fsum1 = _mm_loadu_ps(outptr + 4);
                _fsum2 = _mm_loadu_ps(outptr + 8);
                _fsum3 = _mm_loadu_ps(outptr + 12);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pA1 = _mm_alignr_epi8(_pA0, _pA0, 8);
                    __m128i _pB0 = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
                    __m128i _pB1 = _mm_alignr_epi8(_pB0, _pB0, 4);
#if __AVXVNNIINT8__
                    _sum0 = _mm_dpbssd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm_dpbssd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm_dpbssd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm_dpbssd_epi32(_sum3, _pB1, _pA1);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpbusd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm_comp_dpbusd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm_comp_dpbusd_epi32(_sum3, _pA1, _pB1);
#endif // __AVXVNNIINT8__
                    pA += 16;
                    pB += 8;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift0 = _mm_loadu_si128((const __m128i*)pB);
                    __m128i _a_shift1 = _mm_alignr_epi8(_a_shift0, _a_shift0, 4);
                    _sum0 = _mm_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm_sub_epi32(_sum1, _a_shift1);
                    _sum2 = _mm_sub_epi32(_sum2, _a_shift0);
                    _sum3 = _mm_sub_epi32(_sum3, _a_shift1);
                    pB += 16;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    __m128i _pA0 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB0 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8));
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pA0 = _mm_unpacklo_epi16(_pA16, _pA16);
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB0 = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pA += 4;
                    pB += 2;
                }

                __m128 _descaleA0 = _mm_loadu_ps(pA_descales);
                __m128 _descaleA1 = _mm_shuffle_ps(_descaleA0, _descaleA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128 _descaleB0 = _mm_loadu_ps(pB_descales);
                __m128 _descaleB1 = _mm_shuffle_ps(_descaleB0, _descaleB0, _MM_SHUFFLE(0, 3, 2, 1));
                _fsum0 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum0), _mm_mul_ps(_descaleA0, _descaleB0), _fsum0);
                _fsum1 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum1), _mm_mul_ps(_descaleA0, _descaleB1), _fsum1);
                _fsum2 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum2), _mm_mul_ps(_descaleA1, _descaleB0), _fsum2);
                _fsum3 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum3), _mm_mul_ps(_descaleA1, _descaleB1), _fsum3);
                pA_descales += 4;
                pB_descales += 4;
            }

            _mm_storeu_ps(outptr, _fsum0);
            _mm_storeu_ps(outptr + 4, _fsum1);
            _mm_storeu_ps(outptr + 8, _fsum2);
            _mm_storeu_ps(outptr + 12, _fsum3);
            outptr += 16;
            pB_panel += ((size_t)4 * K + 1) / 2 + (size_t)4 * 4 * correction_block_count;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 2;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 2;
            __m128 _fsum0;
            __m128 _fsum1;

            if (k == 0)
            {
                _fsum0 = _mm_setzero_ps();
                _fsum1 = _mm_setzero_ps();
            }
            else
            {
                _fsum0 = _mm_loadu_ps(outptr);
                _fsum1 = _mm_loadu_ps(outptr + 4);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    __m128i _pB0 = _mm_unpacklo_epi64(_pB8, _pB8);
                    __m128i _pB1 = _mm_alignr_epi8(_pB0, _pB0, 4);
#if __AVXVNNIINT8__
                    _sum0 = _mm_dpbssd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm_dpbssd_epi32(_sum1, _pB1, _pA0);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpbusd_epi32(_sum1, _pA0, _pB1);
#endif // __AVXVNNIINT8__
                    pA += 16;
                    pB += 4;
                }
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift0 = _mm_loadl_epi64((const __m128i*)pB);
                    _a_shift0 = _mm_shuffle_epi32(_a_shift0, _MM_SHUFFLE(1, 0, 1, 0));
                    __m128i _a_shift1 = _mm_alignr_epi8(_a_shift0, _a_shift0, 4);
                    _sum0 = _mm_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm_sub_epi32(_sum1, _a_shift1);
                    pB += 8;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB8 = _mm_shuffle_epi32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8)), _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pA0 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB0 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    pA += 8;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pB8 = _mm_set1_epi16(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]))) & 0xffff);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pA0 = _mm_unpacklo_epi16(_pA16, _pA16);
                    __m128i _pB0 = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    pA += 4;
                    pB += 1;
                }

                __m128 _descaleA = _mm_loadu_ps(pA_descales);
                __m128 _descaleB0 = _mm_setr_ps(pB_descales[0], pB_descales[1], pB_descales[0], pB_descales[1]);
                __m128 _descaleB1 = _mm_shuffle_ps(_descaleB0, _descaleB0, _MM_SHUFFLE(0, 3, 2, 1));
                _fsum0 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum0), _mm_mul_ps(_descaleA, _descaleB0), _fsum0);
                _fsum1 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum1), _mm_mul_ps(_descaleA, _descaleB1), _fsum1);
                pA_descales += 4;
                pB_descales += 2;
            }

            _mm_storeu_ps(outptr, _fsum0);
            _mm_storeu_ps(outptr + 4, _fsum1);
            outptr += 8;
            pB_panel += ((size_t)2 * K + 1) / 2 + (size_t)2 * 4 * correction_block_count;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset;
            const float* pB_descales = pB_descales_panel + (size_t)block_start;
            __m128 _fsum;

            if (k == 0)
            {
                _fsum = _mm_setzero_ps();
            }
            else
            {
                _fsum = _mm_loadu_ps(outptr);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum = _mm_setzero_si128();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = _mm_set1_epi32(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8))));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else // __AVXVNNIINT8__
#if __AVX512VNNI__ && _MSC_VER < 1932
                    // old msvc crash here  --- nihui
                    __m256i _pA0 = _mm256_cvtepu8_epi16(_pA);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);
                    __m256i _s0 = _mm256_madd_epi16(_pA0, _pB0);
                    __m128i _s1 = _mm_hadd_epi32(_mm256_extracti128_si256(_s0, 0), _mm256_extracti128_si256(_s0, 1));
                    _sum = _mm_add_epi32(_sum, _s1);
#else
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif
#endif // __AVXVNNIINT8__
                    pA += 16;
                    pB += 2;
                }
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    _sum = _mm_sub_epi32(_sum, _mm_set1_epi32(_mm_cvtsi128_si32(_mm_castps_si128(_mm_load_ss((const float*)pB)))));
                    pB += 4;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB8 = _mm_set1_epi16(_mm_cvtsi128_si32(decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]))) & 0xffff);
                    __m128i _pA = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 8;
                    pB += 1;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_unpacklo_epi16(_pA16, _mm_setzero_si128());
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _mm_set1_epi16(get_int4_value_wq_int4(pB, 0)));
                    pA += 4;
                    pB++;
                }

                __m128 _descaleA = _mm_loadu_ps(pA_descales);
                _fsum = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum), _mm_mul_ps(_descaleA, _mm_set1_ps(pB_descales[0])), _fsum);
                pA_descales += 4;
                pB_descales++;
            }

            _mm_storeu_ps(outptr, _fsum);
            outptr += 4;
            pB_panel += ((size_t)K + 1) / 2 + (size_t)4 * correction_block_count;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 4;
        pAT_descales += A_descales_hstep * 4;
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 8;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 8;
            __m256 _fsum0;
            __m256 _fsum1;

            if (k == 0)
            {
                _fsum0 = _mm256_setzero_ps();
                _fsum1 = _mm256_setzero_ps();
            }
            else
            {
                _fsum0 = _mm256_loadu_ps(outptr);
                _fsum1 = _mm256_loadu_ps(outptr + 8);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pA128 = _mm_unpacklo_epi64(_pA8, _pA8);
                    __m256i _pA0 = _mm256_broadcastsi128_si256(_pA128);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB = decode_int4x32_wq_int4(_mm_loadu_si128((const __m128i*)pB));
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pA0, _pB);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pA1, _pB);
                    pA += 8;
                    pB += 16;
                }
                if (max_kk0 >= 4)
                {
                    __m256i _a_shift = _mm256_loadu_si256((const __m256i*)pB);
                    _sum0 = _mm256_sub_epi32(_sum0, _a_shift);
                    _sum1 = _mm256_sub_epi32(_sum1, _a_shift);
                    pB += 32;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pA16x1 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA16 = _mm_unpacklo_epi64(_pA16x1, _pA16x1);
                    __m256i _pA0 = _mm256_broadcastsi128_si256(_pA16);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB = _mm256_cvtepi8_epi16(decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB)));
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA1, _pB);
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0] | (unsigned char)pA[1] << 8);
                    __m128i _pA32x1 = _mm_cvtepi8_epi32(_pA8);
                    __m128i _pA128 = _mm_shuffle_epi32(_pA32x1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256i _pA0 = _mm256_broadcastsi128_si256(_pA128);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB = _mm256_cvtepi8_epi32(decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB))));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_mullo_epi32(_pA0, _pB));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_mullo_epi32(_pA1, _pB));
                    pA += 2;
                    pB += 4;
                }

                __m128 _descaleA2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pA_descales);
                __m128 _descaleA128 = _mm_movelh_ps(_descaleA2, _descaleA2);
                __m256 _descaleA0 = combine4x2_ps(_descaleA128, _descaleA128);
                __m256 _descaleA1 = _mm256_shuffle_ps(_descaleA0, _descaleA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256 _descaleB = _mm256_loadu_ps(pB_descales);
                _fsum0 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_descaleA0, _descaleB), _fsum0);
                _fsum1 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_descaleA1, _descaleB), _fsum1);
                pA_descales += 2;
                pB_descales += 8;
            }

            _mm256_storeu_ps(outptr, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            outptr += 16;
            pB_panel += ((size_t)8 * K + 1) / 2 + (size_t)8 * 4 * correction_block_count;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __AVX512F__
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 4;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 4;
            __m128 _fsum0;
            __m128 _fsum1;

            if (k == 0)
            {
                _fsum0 = _mm_setzero_ps();
                _fsum1 = _mm_setzero_ps();
            }
            else
            {
                _fsum0 = _mm_loadu_ps(outptr);
                _fsum1 = _mm_loadu_ps(outptr + 4);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pA = _mm_unpacklo_epi64(_pA8, _pA8);
                    __m128i _pB0 = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
#if __AVXVNNIINT8__
                    _sum0 = _mm_dpbssd_epi32(_sum0, _pB0, _pA);
                    _sum1 = _mm_dpbssd_epi32(_sum1, _pB1, _pA);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pA, _pB0);
                    _sum1 = _mm_comp_dpbusd_epi32(_sum1, _pA, _pB1);
#endif // __AVXVNNIINT8__
                    pA += 8;
                    pB += 8;
                }
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift0 = _mm_loadu_si128((const __m128i*)pB);
                    __m128i _a_shift1 = _mm_shuffle_epi32(_a_shift0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_sub_epi32(_sum0, _a_shift0);
                    _sum1 = _mm_sub_epi32(_sum1, _a_shift1);
                    pB += 16;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pA16x1 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_unpacklo_epi64(_pA16x1, _pA16x1);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    __m128i _pB0 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA, _pB1);
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0] | (unsigned char)pA[1] << 8);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA32x1 = _mm_unpacklo_epi16(_pA16, _pA16);
                    __m128i _pA = _mm_shuffle_epi32(_pA32x1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8));
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB0 = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA, _pB1);
                    pA += 2;
                    pB += 2;
                }

                __m128 _descaleA2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pA_descales);
                __m128 _descaleA = _mm_movelh_ps(_descaleA2, _descaleA2);
                __m128 _descaleB0 = _mm_loadu_ps(pB_descales);
                __m128 _descaleB1 = _mm_shuffle_ps(_descaleB0, _descaleB0, _MM_SHUFFLE(0, 3, 2, 1));
                _fsum0 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum0), _mm_mul_ps(_descaleA, _descaleB0), _fsum0);
                _fsum1 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum1), _mm_mul_ps(_descaleA, _descaleB1), _fsum1);
                pA_descales += 2;
                pB_descales += 4;
            }

            _mm_storeu_ps(outptr, _fsum0);
            _mm_storeu_ps(outptr + 4, _fsum1);
            outptr += 8;
            pB_panel += ((size_t)4 * K + 1) / 2 + (size_t)4 * 4 * correction_block_count;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 2;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 2;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
#if __SSE2__
            __m128 _fsum;

            if (k == 0)
            {
                _fsum = _mm_setzero_ps();
            }
            else
            {
                _fsum = _mm_loadu_ps(outptr);
            }
#else  // __SSE2__
            float fsum00;
            float fsum01;
            float fsum10;
            float fsum11;

            if (k == 0)
            {
                fsum00 = 0.f;
                fsum01 = 0.f;
                fsum10 = 0.f;
                fsum11 = 0.f;
            }
            else
            {
                fsum00 = outptr[0];
                fsum01 = outptr[1];
                fsum10 = outptr[2];
                fsum11 = outptr[3];
            }
#endif // __SSE2__
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __SSE2__
                __m128i _sum = _mm_setzero_si128();
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pA = _mm_unpacklo_epi32(_pA8, _pA8);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    __m128i _pB = _mm_unpacklo_epi64(_pB8, _pB8);
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif // __AVXVNNIINT8__
                    pA += 8;
                    pB += 4;
                }
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift = _mm_loadl_epi64((const __m128i*)pB);
                    _a_shift = _mm_shuffle_epi32(_a_shift, _MM_SHUFFLE(1, 0, 1, 0));
                    _sum = _mm_sub_epi32(_sum, _a_shift);
                    pB += 8;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pA16x1 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_unpacklo_epi32(_pA16x1, _pA16x1);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8));
                    __m128i _pB16x1 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB = _mm_unpacklo_epi64(_pB16x1, _pB16x1);
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0] | (unsigned char)pA[1] << 8);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA32x1 = _mm_unpacklo_epi16(_pA16, _pA16);
                    __m128i _pA = _mm_unpacklo_epi32(_pA32x1, _pA32x1);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]));
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB32x1 = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    __m128i _pB = _mm_unpacklo_epi64(_pB32x1, _pB32x1);
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 1;
                }

                __m128 _descaleA2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pA_descales);
                __m128 _descaleA = _mm_unpacklo_ps(_descaleA2, _descaleA2);
                __m128 _descaleB2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pB_descales);
                __m128 _descaleB = _mm_movelh_ps(_descaleB2, _descaleB2);
                _fsum = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum), _mm_mul_ps(_descaleA, _descaleB), _fsum);
#else  // __SSE2__
                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;
                for (; kk < max_kk0; kk++)
                {
                    const int b0 = get_int4_value_wq_int4(pB, 0);
                    const int b1 = get_int4_value_wq_int4(pB, 1);
                    sum00 += pA[0] * b0;
                    sum01 += pA[0] * b1;
                    sum10 += pA[1] * b0;
                    sum11 += pA[1] * b1;
                    pA += 2;
                    pB += 1;
                }

                const float ad0 = pA_descales[0];
                const float ad1 = pA_descales[1];
                fsum00 += sum00 * ad0 * pB_descales[0];
                fsum01 += sum01 * ad0 * pB_descales[1];
                fsum10 += sum10 * ad1 * pB_descales[0];
                fsum11 += sum11 * ad1 * pB_descales[1];
#endif // __SSE2__
                pA_descales += 2;
                pB_descales += 2;
            }

#if __SSE2__
            _mm_storeu_ps(outptr, _fsum);
#else  // __SSE2__
            outptr[0] = fsum00;
            outptr[1] = fsum01;
            outptr[2] = fsum10;
            outptr[3] = fsum11;
#endif // __SSE2__
            outptr += 4;
            pB_panel += ((size_t)2 * K + 1) / 2 + (size_t)2 * 4 * correction_block_count;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset;
            const float* pB_descales = pB_descales_panel + (size_t)block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float fsum0;
            float fsum1;

            if (k == 0)
            {
                fsum0 = 0.f;
                fsum1 = 0.f;
            }
            else
            {
                fsum0 = outptr[0];
                fsum1 = outptr[1];
            }
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum0 = 0;
                int sum1 = 0;
                int kk = 0;
#if __SSE2__
                __m128i _sum = _mm_setzero_si128();
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    _pB = _mm_unpacklo_epi32(_pB, _pB);
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif // __AVXVNNIINT8__
                    pA += 16;
                    pB += 4;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8));
                    _pB = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 0, 0, 0));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif // __AVXVNNIINT8__
                    pA += 8;
                    pB += 2;
                }
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift = _mm_castps_si128(_mm_load_ss((const float*)pB));
                    _a_shift = _mm_shuffle_epi32(_a_shift, _MM_SHUFFLE(0, 0, 0, 0));
                    _a_shift = _mm_unpacklo_epi64(_a_shift, _mm_setzero_si128());
                    _sum = _mm_sub_epi32(_sum, _a_shift);
                    pB += 4;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pA = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _pB = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 0, 0, 0));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 4;
                    pB += 1;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0] | (unsigned char)pA[1] << 8);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_unpacklo_epi16(_pA16, _mm_setzero_si128());
                    __m128i _pB = _mm_set1_epi32((unsigned short)get_int4_value_wq_int4(pB, 0));
                    _sum = _mm_add_epi32(_sum, _mm_madd_epi16(_pA, _pB));
                    pA += 2;
                    pB++;
                }

                _sum = _mm_add_epi32(_sum, _mm_shuffle_epi32(_sum, _MM_SHUFFLE(3, 2, 3, 2)));
                sum0 = _mm_cvtsi128_si32(_sum);
                sum1 = _mm_cvtsi128_si32(_mm_shuffle_epi32(_sum, _MM_SHUFFLE(1, 1, 1, 1)));
#else  // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    const int b0 = get_int4_value_wq_int4(pB, kk & 1);
                    sum0 += pA[0] * b0;
                    sum1 += pA[1] * b0;
                    pA += 2;
                    if (kk & 1)
                        pB++;
                }
                if (max_kk0 & 1)
                    pB++;
#endif // __SSE2__

                fsum0 += sum0 * pA_descales[0] * pB_descales[0];
                fsum1 += sum1 * pA_descales[1] * pB_descales[0];
                pA_descales += 2;
                pB_descales++;
            }

            outptr[0] = fsum0;
            outptr[1] = fsum1;
            outptr += 2;
            pB_panel += ((size_t)K + 1) / 2 + (size_t)4 * correction_block_count;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 8;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 8;
            __m256 _fsum;

            if (k == 0)
            {
                _fsum = _mm256_setzero_ps();
            }
            else
            {
                _fsum = _mm256_loadu_ps(outptr);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum = _mm256_setzero_si256();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA32 = _mm_castps_si128(_mm_load_ss((const float*)pA));
#if defined(_MSC_VER) && _MSC_VER < 1930
                    // old msvc crash here  --- nihui
                    __m256i _pA = _mm256_broadcastsi128_si256(_mm_shuffle_epi32(_pA32, _MM_SHUFFLE(0, 0, 0, 0)));
#else
                    __m256i _pA = _mm256_broadcastd_epi32(_pA32);
#endif
                    __m256i _pB = decode_int4x32_wq_int4(_mm_loadu_si128((const __m128i*)pB));
                    _sum = _mm256_comp_dpbusd_epi32(_sum, _pA, _pB);
                    pA += 4;
                    pB += 16;
                }
                if (max_kk0 >= 4)
                {
                    __m256i _a_shift = _mm256_loadu_si256((const __m256i*)pB);
                    _sum = _mm256_sub_epi32(_sum, _a_shift);
                    pB += 32;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0] | (unsigned char)pA[1] << 8);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m256i _pA = _mm256_broadcastsi128_si256(_mm_shuffle_epi32(_pA16, _MM_SHUFFLE(0, 0, 0, 0)));
                    __m256i _pB = _mm256_cvtepi8_epi16(decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB)));
                    _sum = _mm256_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128(pA[0]);
                    __m256i _pA = _mm256_broadcastd_epi32(_pA8);
                    __m256i _pB = _mm256_cvtepi8_epi32(decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB))));
                    _sum = _mm256_add_epi32(_sum, _mm256_mullo_epi32(_pA, _pB));
                    pA++;
                    pB += 4;
                }
                __m128 _descaleA1 = _mm_load_ss(pA_descales);
                __m256 _descaleA = _mm256_broadcastss_ps(_descaleA1);
                __m256 _descale = _mm256_mul_ps(_descaleA, _mm256_loadu_ps(pB_descales));
                _fsum = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_sum), _descale, _fsum);
                pA_descales += 1;
                pB_descales += 8;
            }

            _mm256_storeu_ps(outptr, _fsum);
            outptr += 8;
            pB_panel += ((size_t)8 * K + 1) / 2 + (size_t)8 * 4 * correction_block_count;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __AVX512F__
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 4;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 4;
            __m128 _fsum;

            if (k == 0)
            {
                _fsum = _mm_setzero_ps();
            }
            else
            {
                _fsum = _mm_loadu_ps(outptr);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum = _mm_setzero_si128();
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA32 = _mm_castps_si128(_mm_load_ss((const float*)pA));
                    __m128i _pA = _mm_shuffle_epi32(_pA32, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif // __AVXVNNIINT8__
                    pA += 4;
                    pB += 8;
                }
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift = _mm_loadu_si128((const __m128i*)pB);
                    _sum = _mm_sub_epi32(_sum, _a_shift);
                    pB += 16;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0] | (unsigned char)pA[1] << 8);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_shuffle_epi32(_pA16, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB8 = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128(pA[0]);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_shuffle_epi32(_mm_unpacklo_epi16(_pA16, _pA16), _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8));
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA++;
                    pB += 2;
                }
                __m128 _descaleA1 = _mm_load_ss(pA_descales);
                __m128 _descaleA = _mm_shuffle_ps(_descaleA1, _descaleA1, _MM_SHUFFLE(0, 0, 0, 0));
                __m128 _descale = _mm_mul_ps(_descaleA, _mm_loadu_ps(pB_descales));
                _fsum = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_sum), _descale, _fsum);
                pA_descales += 1;
                pB_descales += 4;
            }

            _mm_storeu_ps(outptr, _fsum);
            outptr += 4;
            pB_panel += ((size_t)4 * K + 1) / 2 + (size_t)4 * 4 * correction_block_count;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset * 2;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 2;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float fsum0;
            float fsum1;

            if (k == 0)
            {
                fsum0 = 0.f;
                fsum1 = 0.f;
            }
            else
            {
                fsum0 = outptr[0];
                fsum1 = outptr[1];
            }
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum0 = 0;
                int sum1 = 0;
                int kk = 0;
#if __SSE2__
                __m128i _sum = _mm_setzero_si128();
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                    _pA = _mm_unpacklo_epi32(_pA, _pA);
                    __m128i _pB = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif // __AVXVNNIINT8__
                    pA += 8;
                    pB += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = _mm_shuffle_epi32(_mm_castps_si128(_mm_load_ss((const float*)pA)), _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB = decode_int4_wq_int4(_mm_castps_si128(_mm_load_ss((const float*)pB)));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif // __AVXVNNIINT8__
                    pA += 4;
                    pB += 4;
                }
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    __m128i _a_shift = _mm_loadl_epi64((const __m128i*)pB);
                    _sum = _mm_sub_epi32(_sum, _a_shift);
                    pB += 8;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0] | (unsigned char)pA[1] << 8);
                    __m128i _pA = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    _pA = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_set1_epi16(pA[0]);
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                    __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                    _sum = _mm_add_epi32(_sum, _mm_unpacklo_epi16(_sl, _sh));
                    pA++;
                    pB += 1;
                }

                _sum = _mm_add_epi32(_sum, _mm_shuffle_epi32(_sum, _MM_SHUFFLE(3, 2, 3, 2)));
                sum0 = _mm_cvtsi128_si32(_sum);
                sum1 = _mm_cvtsi128_si32(_mm_shuffle_epi32(_sum, _MM_SHUFFLE(1, 1, 1, 1)));
#else  // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * get_int4_value_wq_int4(pB, 0);
                    sum1 += pA[0] * get_int4_value_wq_int4(pB, 1);
                    pA++;
                    pB += 1;
                }
#endif // __SSE2__

                const float ad = pA_descales[0];
                fsum0 += sum0 * ad * pB_descales[0];
                fsum1 += sum1 * ad * pB_descales[1];
                pA_descales += 1;
                pB_descales += 2;
            }

            outptr[0] = fsum0;
            outptr[1] = fsum1;
            outptr += 2;
            pB_panel += ((size_t)2 * K + 1) / 2 + (size_t)2 * 4 * correction_block_count;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + (size_t)b_offset;
            const float* pB_descales = pB_descales_panel + (size_t)block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float fsum;

            if (k == 0)
            {
                fsum = 0.f;
            }
            else
            {
                fsum = outptr[0];
            }
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum = 0;
                int kk = 0;
#if __SSE2__
                __m128i _sum = _mm_setzero_si128();
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = decode_int4_wq_int4(_mm_loadl_epi64((const __m128i*)pB));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif // __AVXVNNIINT8__
                    pA += 16;
                    pB += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = _mm_shuffle_epi32(_mm_castps_si128(_mm_load_ss((const float*)pA)), _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0] | (unsigned char)pB[1] << 8));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pA, _pB);
#endif // __AVXVNNIINT8__
                    pA += 4;
                    pB += 2;
                }
#if !__AVXVNNIINT8__
                if (max_kk0 >= 4)
                {
                    _sum = _mm_sub_epi32(_sum, _mm_castps_si128(_mm_load_ss((const float*)pB)));
                    pB += 4;
                }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0] | (unsigned char)pA[1] << 8);
                    __m128i _pA = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB8 = decode_int4_wq_int4(_mm_cvtsi32_si128((unsigned char)pB[0]));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 1;
                }

                sum = _mm_reduce_add_epi32(_sum);
                for (; kk < max_kk0; kk++)
                {
                    sum += pA[0] * get_int4_value_wq_int4(pB, 0);
                    pA++;
                    pB++;
                }
#else  // __SSE2__
                for (; kk < max_kk0; kk++)
                {
                    sum += pA[0] * get_int4_value_wq_int4(pB, kk & 1);
                    pA++;
                    if (kk & 1)
                        pB++;
                }
                if (max_kk0 & 1)
                    pB++;
#endif // __SSE2__

                fsum += sum * pA_descales[0] * pB_descales[0];
                pA_descales += 1;
                pB_descales++;
            }
            outptr[0] = fsum;
            outptr++;
            pB_panel += ((size_t)K + 1) / 2 + (size_t)4 * correction_block_count;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void get_optimal_tile_mnk_wq_int4(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // A is int8, B is packed int4, and each block carries A/B descales plus
    // the worst-case VNNI correction payload for B.
    const float bytes_per_k = 1.f + .5f + 4.f + 12.f / block_size;
    int tile_size = (int)sqrtf((float)l2_cache_size / bytes_per_k);

#if __AVX512F__
    TILE_M = std::max(16, tile_size / 16 * 16);
#elif __AVX__
    TILE_M = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
    TILE_M = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
#endif // __AVX512F__

#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    TILE_N = std::max(8, tile_size / 8 * 8);
#else
    TILE_N = std::max(4, tile_size / 4 * 4);
#endif // __AVX512F__
#else
    TILE_N = std::max(2, tile_size / 2 * 2);
#endif // defined(__x86_64__) || defined(_M_X64)

    TILE_K = std::max(block_size, tile_size / block_size * block_size);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(K, ((K + nn_K - 1) / nn_K + block_size - 1) / block_size * block_size);

        if (nn_K == 1)
        {
            const float packed_ab_bytes_per_k = 1.f + .5f + 12.f / block_size;
            tile_size = std::max(1, (int)((float)l2_cache_size / packed_ab_bytes_per_k / TILE_K));

#if __AVX512F__
            TILE_M = std::max(16, tile_size / 16 * 16);
#elif __AVX__
            TILE_M = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
            TILE_M = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
#endif // __AVX512F__

#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
            TILE_N = std::max(8, tile_size / 8 * 8);
#else
            TILE_N = std::max(4, tile_size / 4 * 4);
#endif // __AVX512F__
#else
            TILE_N = std::max(2, tile_size / 2 * 2);
#endif // defined(__x86_64__) || defined(_M_X64)
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __AVX512F__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif // __AVX512F__
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#endif // __AVX512F__
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2);
#endif // defined(__x86_64__) || defined(_M_X64)
    }

    if (nT > 1)
    {
#if __AVX512F__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif // __AVX512F__
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __AVX512F__
        TILE_M = (constant_TILE_M + 15) / 16 * 16;
#elif __AVX__
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#elif __SSE2__
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif // __AVX512F__
    }

    if (constant_TILE_N > 0)
    {
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#else
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#endif // __AVX512F__
#else
        TILE_N = (constant_TILE_N + 1) / 2 * 2;
#endif // defined(__x86_64__) || defined(_M_X64)
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, (constant_TILE_K + block_size - 1) / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(K, TILE_K);
    }
}

// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static inline uint32_t unpack_int4x4(const unsigned char* p)
{
    const uint32_t v = p[0] | ((uint32_t)p[1] << 8);
    const uint32_t q = (v & 0xf) | ((v & 0xf0) << 4) | ((v & 0xf00) << 8) | ((v & 0xf000) << 12);
    return q | ((q & 0x08080808) * 30);
}

static void pack_B_tile_int4s(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int j, int max_jj, int K)
{
    // Keep the reference FP32 division with fast math enabled.
    int jj = 0;
#if __SSE2__
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 8;

        for (int jj0 = 0; jj0 < 16; jj0 += 4)
        {
            const unsigned char* p0 = B.row<const unsigned char>(j + jj + jj0);
            const unsigned char* p1 = B.row<const unsigned char>(j + jj + jj0 + 1);
            const unsigned char* p2 = B.row<const unsigned char>(j + jj + jj0 + 2);
            const unsigned char* p3 = B.row<const unsigned char>(j + jj + jj0 + 3);
            unsigned char* pp = pp0 + jj0 / 2;

            int kk = 0;
            for (; kk + 1 < K; kk += 2)
            {
                const unsigned int v0 = *p0++;
                const unsigned int v1 = *p1++;
                const unsigned int v2 = *p2++;
                const unsigned int v3 = *p3++;
                pp[0] = (v0 & 15) | (v1 << 4);
                pp[1] = (v2 & 15) | (v3 << 4);
                pp[bytes] = (v0 >> 4) | (v1 & 0xf0);
                pp[bytes + 1] = (v2 >> 4) | (v3 & 0xf0);
                pp += bytes * 2;
            }
            if (kk < K)
            {
                pp[0] = (*p0 & 15) | (*p1 << 4);
                pp[1] = (*p2 & 15) | (*p3 << 4);
            }
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 16; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 16 + jj0] = 1.f / scales[g];
        }
    }
#else  // __AVX512F__
    for (; jj + 11 < max_jj; jj += 12)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 6;

        for (int jj0 = 0; jj0 < 12; jj0 += 4)
        {
            const unsigned char* p0 = B.row<const unsigned char>(j + jj + jj0);
            const unsigned char* p1 = B.row<const unsigned char>(j + jj + jj0 + 1);
            const unsigned char* p2 = B.row<const unsigned char>(j + jj + jj0 + 2);
            const unsigned char* p3 = B.row<const unsigned char>(j + jj + jj0 + 3);
            unsigned char* pp = pp0 + jj0 / 2;

            int kk = 0;
            for (; kk + 1 < K; kk += 2)
            {
                const unsigned int v0 = *p0++;
                const unsigned int v1 = *p1++;
                const unsigned int v2 = *p2++;
                const unsigned int v3 = *p3++;
                pp[0] = (v0 & 15) | (v1 << 4);
                pp[1] = (v2 & 15) | (v3 << 4);
                pp[bytes] = (v0 >> 4) | (v1 & 0xf0);
                pp[bytes + 1] = (v2 >> 4) | (v3 & 0xf0);
                pp += bytes * 2;
            }
            if (kk < K)
            {
                pp[0] = (*p0 & 15) | (*p1 << 4);
                pp[1] = (*p2 & 15) | (*p3 << 4);
            }
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 12; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 12 + jj0] = 1.f / scales[g];
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 4;

        for (int jj0 = 0; jj0 < 8; jj0 += 4)
        {
            const unsigned char* p0 = B.row<const unsigned char>(j + jj + jj0);
            const unsigned char* p1 = B.row<const unsigned char>(j + jj + jj0 + 1);
            const unsigned char* p2 = B.row<const unsigned char>(j + jj + jj0 + 2);
            const unsigned char* p3 = B.row<const unsigned char>(j + jj + jj0 + 3);
            unsigned char* pp = pp0 + jj0 / 2;

            int kk = 0;
            for (; kk + 1 < K; kk += 2)
            {
                const unsigned int v0 = *p0++;
                const unsigned int v1 = *p1++;
                const unsigned int v2 = *p2++;
                const unsigned int v3 = *p3++;
                pp[0] = (v0 & 15) | (v1 << 4);
                pp[1] = (v2 & 15) | (v3 << 4);
                pp[bytes] = (v0 >> 4) | (v1 & 0xf0);
                pp[bytes + 1] = (v2 >> 4) | (v3 & 0xf0);
                pp += bytes * 2;
            }
            if (kk < K)
            {
                pp[0] = (*p0 & 15) | (*p1 << 4);
                pp[1] = (*p2 & 15) | (*p3 << 4);
            }
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 8; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 8 + jj0] = 1.f / scales[g];
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
        unsigned char* pp = BT.row<unsigned char>(j + jj);
        const int bytes = 2;

        int kk = 0;
        for (; kk + 1 < K; kk += 2)
        {
            const unsigned int v0 = *p0++;
            const unsigned int v1 = *p1++;
            const unsigned int v2 = *p2++;
            const unsigned int v3 = *p3++;
            pp[0] = (v0 & 15) | (v1 << 4);
            pp[1] = (v2 & 15) | (v3 << 4);
            pp[bytes] = (v0 >> 4) | (v1 & 0xf0);
            pp[bytes + 1] = (v2 >> 4) | (v3 & 0xf0);
            pp += bytes * 2;
        }
        if (kk < K)
        {
            pp[0] = (*p0 & 15) | (*p1 << 4);
            pp[1] = (*p2 & 15) | (*p3 << 4);
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 4; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 4 + jj0] = 1.f / scales[g];
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        unsigned char* pp = BT.row<unsigned char>(j + jj);

        int kk = 0;
        for (; kk + 1 < K; kk += 2)
        {
            const unsigned int v0 = *p0++;
            const unsigned int v1 = *p1++;
            pp[0] = (v0 & 15) | (v1 << 4);
            pp[1] = (v0 >> 4) | (v1 & 0xf0);
            pp += 2;
        }
        if (kk < K)
            pp[0] = (*p0 & 15) | (*p1 << 4);

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 2; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 2 + jj0] = 1.f / scales[g];
        }
    }
    for (; jj < max_jj; jj++)
    {
        if (BT.data != B.data)
            memcpy(BT.row<unsigned char>(j + jj), B.row<const unsigned char>(j + jj), B.w);

        float* descales = BT_descales.row(j + jj);
        const volatile float* scales = B_scales.row(j + jj);
        for (int g = 0; g < BT_descales.w; g++)
            descales[g] = 1.f / scales[g];
    }
}

#if __AVX512F__
static NCNN_FORCEINLINE __m128i unpack16_int4(const unsigned char* ptr)
{
    __m128i _v = _mm_loadl_epi64((const __m128i*)ptr);
    __m128i _mask = _mm_set1_epi8(15);
    __m128i _sign = _mm_set1_epi8(8);
    _v = _mm_unpacklo_epi8(_mm_and_si128(_v, _mask), _mm_and_si128(_mm_srli_epi16(_v, 4), _mask));
    return _mm_sub_epi8(_mm_xor_si128(_v, _sign), _sign);
}
#endif // __AVX512F__

static void dequantize_B_tile_int4s(const Mat& B, const Mat& B_scales, Mat& BT_tile, int j, int max_jj, int k, int max_kk, int block_size)
{
    const int k_end = k + max_kk;
    const int g0 = k / block_size;
    const int g_end = (k_end - 1) / block_size + 1;

    float* pp = BT_tile;
    int jj = 0;
#if __SSE2__
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 16;

            const unsigned char* p = pB + (size_t)k0 * 8;
            const __m512 _descale = _mm512_loadu_ps(descales);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const __m128i _v = unpack16_int4(p);
                const __m512 _f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_v));
                _mm512_storeu_ps(pp, _mm512_mul_ps(_f, _descale));
                p += 8;
                pp += 16;
            }
        }
    }
#else // __AVX512F__
    for (; jj + 11 < max_jj; jj += 12)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 12;

            const unsigned char* p = pB + (size_t)k0 * 6;
            const __m128 _descale0 = _mm_loadu_ps(descales);
            const __m128 _descale1 = _mm_loadu_ps(descales + 4);
            const __m128 _descale2 = _mm_loadu_ps(descales + 8);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const uint32_t v0 = unpack_int4x4(p);
                const uint32_t v1 = unpack_int4x4(p + 2);
                const uint32_t v2 = unpack_int4x4(p + 4);
                __m128i _v0 = _mm_cvtsi32_si128(v0);
                __m128i _v1 = _mm_cvtsi32_si128(v1);
                __m128i _v2 = _mm_cvtsi32_si128(v2);
#if __SSE4_1__
                _v0 = _mm_cvtepi8_epi32(_v0);
                _v1 = _mm_cvtepi8_epi32(_v1);
                _v2 = _mm_cvtepi8_epi32(_v2);
#else
                const __m128i _zero = _mm_setzero_si128();
                _v0 = _mm_unpacklo_epi8(_v0, _mm_cmpgt_epi8(_zero, _v0));
                _v0 = _mm_unpacklo_epi16(_v0, _mm_srai_epi16(_v0, 15));
                _v1 = _mm_unpacklo_epi8(_v1, _mm_cmpgt_epi8(_zero, _v1));
                _v1 = _mm_unpacklo_epi16(_v1, _mm_srai_epi16(_v1, 15));
                _v2 = _mm_unpacklo_epi8(_v2, _mm_cmpgt_epi8(_zero, _v2));
                _v2 = _mm_unpacklo_epi16(_v2, _mm_srai_epi16(_v2, 15));
#endif // __SSE4_1__
                _mm_storeu_ps(pp, _mm_mul_ps(_mm_cvtepi32_ps(_v0), _descale0));
                _mm_storeu_ps(pp + 4, _mm_mul_ps(_mm_cvtepi32_ps(_v1), _descale1));
                _mm_storeu_ps(pp + 8, _mm_mul_ps(_mm_cvtepi32_ps(_v2), _descale2));
                p += 6;
                pp += 12;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 8;

            const unsigned char* p = pB + (size_t)k0 * 4;
            const __m128 _descale0 = _mm_loadu_ps(descales);
            const __m128 _descale1 = _mm_loadu_ps(descales + 4);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const uint32_t v0 = unpack_int4x4(p);
                const uint32_t v1 = unpack_int4x4(p + 2);
                __m128i _v0 = _mm_cvtsi32_si128(v0);
                __m128i _v1 = _mm_cvtsi32_si128(v1);
#if __SSE4_1__
                _v0 = _mm_cvtepi8_epi32(_v0);
                _v1 = _mm_cvtepi8_epi32(_v1);
#else
                const __m128i _zero = _mm_setzero_si128();
                _v0 = _mm_unpacklo_epi8(_v0, _mm_cmpgt_epi8(_zero, _v0));
                _v0 = _mm_unpacklo_epi16(_v0, _mm_srai_epi16(_v0, 15));
                _v1 = _mm_unpacklo_epi8(_v1, _mm_cmpgt_epi8(_zero, _v1));
                _v1 = _mm_unpacklo_epi16(_v1, _mm_srai_epi16(_v1, 15));
#endif // __SSE4_1__
                _mm_storeu_ps(pp, _mm_mul_ps(_mm_cvtepi32_ps(_v0), _descale0));
                _mm_storeu_ps(pp + 4, _mm_mul_ps(_mm_cvtepi32_ps(_v1), _descale1));
                p += 4;
                pp += 8;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 4;

            const unsigned char* p = pB + (size_t)k0 * 2;
            const __m128 _descale = _mm_loadu_ps(descales);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const uint32_t v = unpack_int4x4(p);
                __m128i _v = _mm_cvtsi32_si128(v);
#if __SSE4_1__
                _v = _mm_cvtepi8_epi32(_v);
#else
                const __m128i _zero = _mm_setzero_si128();
                _v = _mm_unpacklo_epi8(_v, _mm_cmpgt_epi8(_zero, _v));
                _v = _mm_unpacklo_epi16(_v, _mm_srai_epi16(_v, 15));
#endif // __SSE4_1__
                _mm_storeu_ps(pp, _mm_mul_ps(_mm_cvtepi32_ps(_v), _descale));
                p += 2;
                pp += 4;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 2;

            const unsigned char* p = pB + k0;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const int v = *p++;
                pp[0] = (((v & 15) ^ 8) - 8) * descales[0];
                pp[1] = (((v >> 4) ^ 8) - 8) * descales[1];
                pp += 2;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float descale = scales[g];

            for (int kk = 0; kk < max_kk0; kk++)
            {
                const int index = k0 + kk;
                const int v = (pB[index / 2] >> (index % 2 * 4)) & 15;
                *pp++ = ((v ^ 8) - 8) * descale;
            }
        }
    }
}

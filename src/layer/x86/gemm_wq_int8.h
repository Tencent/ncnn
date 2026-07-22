// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void pack_B_tile_wq_int8_avx512vnni(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_avx512vnni(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void transpose_quantize_A_tile_wq_int8_avx512vnni(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void gemm_transB_packed_tile_wq_int8_avx512vnni(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
void unpack_output_tile_wq_int8_avx512vnni(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
void transpose_unpack_output_tile_wq_int8_avx512vnni(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_B_tile_wq_int8_avxvnniint8(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_avxvnniint8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void transpose_quantize_A_tile_wq_int8_avxvnniint8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void gemm_transB_packed_tile_wq_int8_avxvnniint8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
void unpack_output_tile_wq_int8_avxvnniint8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
void transpose_unpack_output_tile_wq_int8_avxvnniint8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_B_tile_wq_int8_avxvnni(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_avxvnni(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void transpose_quantize_A_tile_wq_int8_avxvnni(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void gemm_transB_packed_tile_wq_int8_avxvnni(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
void unpack_output_tile_wq_int8_avxvnni(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
void transpose_unpack_output_tile_wq_int8_avxvnni(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_B_tile_wq_int8_avx2(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_avx2(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void transpose_quantize_A_tile_wq_int8_avx2(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void gemm_transB_packed_tile_wq_int8_avx2(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
void unpack_output_tile_wq_int8_avx2(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
void transpose_unpack_output_tile_wq_int8_avx2(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void gemm_transB_packed_tile_wq_int8_xop(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

static void pack_B_tile_wq_int8(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_B_tile_wq_int8_avx512vnni(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        pack_B_tile_wq_int8_avxvnniint8(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_B_tile_wq_int8_avxvnni(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_B_tile_wq_int8_avx2(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

    const int block_count = (K + block_size - 1) / block_size;
    unsigned char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const float* ps = B_scales.row(j + jj);

        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(B.w));
        __m256i _sindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _sindex = _mm256_mullo_epi32(_sindex, _mm256_set1_epi32(B_scales.w));

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _p = _mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
                _mm256_storeu_si256((__m256i*)pp, _p);
                pp += 32;
                p0 += 4;
            }
#else  // __AVXVNNIINT8__
            __m256i _v127 = _mm256_set1_epi8(127);
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _p = _mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
                _p = _mm256_add_epi8(_p, _v127);
                _mm256_storeu_si256((__m256i*)pp, _p);
                pp += 32;
                p0 += 4;
            }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _p = _mm256_comp_cvtepi32_epi16(_mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
                _mm_storeu_si128((__m128i*)pp, _p);
                pp += 16;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                __m128i _p = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
                _mm_storel_epi64((__m128i*)pp, _p);
                pp += 8;
                p0++;
            }

            __m256 _scale = _mm256_i32gather_ps(ps, _sindex, sizeof(float));
            _mm256_storeu_ps(pd, _mm256_div_ps(_mm256_set1_ps(1.f), _scale));
            pd += 8;
            ps++;
        }
    }
#endif // __AVX512F__
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const signed char* p1 = B.row<const signed char>(j + jj + 1);
        const signed char* p2 = B.row<const signed char>(j + jj + 2);
        const signed char* p3 = B.row<const signed char>(j + jj + 3);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            // VNNI consumes one contiguous K4 dword per output lane.
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _p = _mm_setr_epi32(_mm_cvtsi128_si32(_mm_loadu_si32(p0)), _mm_cvtsi128_si32(_mm_loadu_si32(p1)), _mm_cvtsi128_si32(_mm_loadu_si32(p2)), _mm_cvtsi128_si32(_mm_loadu_si32(p3)));
#if !__AVXVNNIINT8__
                _p = _mm_add_epi8(_p, _mm_set1_epi8(127));
#endif // __AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _p);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
#else
            // AVX2/SSE2 consumes two K2 vectors for each real K4 region.
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _p01 = _mm_setr_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(p0)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p1)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p2)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p3)), 0, 0, 0, 0);
                __m128i _p23 = _mm_setr_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(p0 + 2)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p1 + 2)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p2 + 2)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p3 + 2)), 0, 0, 0, 0);
                _mm_storel_epi64((__m128i*)pp, _p01);
                _mm_storel_epi64((__m128i*)(pp + 8), _p23);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__ \
// K2/K1 are always signed and compact, including classic VNNI.
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _p = _mm_setr_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(p0)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p1)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p2)), (short)_mm_cvtsi128_si32(_mm_loadu_si16(p3)), 0, 0, 0, 0);
                _mm_storel_epi64((__m128i*)pp, _p);
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = (unsigned char)*p0++;
                pp[1] = (unsigned char)*p1++;
                pp[2] = (unsigned char)*p2++;
                pp[3] = (unsigned char)*p3++;
                pp += 4;
            }

            pd[0] = 1.f / *ps0++;
            pd[1] = 1.f / *ps1++;
            pd[2] = 1.f / *ps2++;
            pd[3] = 1.f / *ps3++;
            pd += 4;
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const signed char* p1 = B.row<const signed char>(j + jj + 1);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            // VNNI consumes one contiguous K4 dword per output lane.
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _p = _mm_setr_epi32(_mm_cvtsi128_si32(_mm_loadu_si32(p0)), _mm_cvtsi128_si32(_mm_loadu_si32(p1)), 0, 0);
#if !__AVXVNNIINT8__
                _p = _mm_add_epi8(_p, _mm_set1_epi8(127));
#endif // __AVXVNNIINT8__
                _mm_storel_epi64((__m128i*)pp, _p);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#else
#if __SSE2__
            // AVX2/SSE2 consumes two K2 vectors for each real K4 region.
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storeu_si16(pp, _mm_loadu_si16(p0));
                _mm_storeu_si16(pp + 2, _mm_loadu_si16(p1));
                _mm_storeu_si16(pp + 4, _mm_loadu_si16(p0 + 2));
                _mm_storeu_si16(pp + 6, _mm_loadu_si16(p1 + 2));
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __SSE2__
#endif // __AVX512VNNI__ || __AVXVNNI__ \
// K2/K1 are always signed and compact, including classic VNNI.
#if __SSE2__
            for (; kk + 1 < max_kk; kk += 2)
            {
                _mm_storeu_si16(pp, _mm_loadu_si16(p0));
                _mm_storeu_si16(pp + 2, _mm_loadu_si16(p1));
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = (unsigned char)*p0++;
                pp[1] = (unsigned char)*p1++;
                pp += 2;
            }

            pd[0] = 1.f / *ps0++;
            pd[1] = 1.f / *ps1++;
            pd += 2;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            // VNNI consumes one contiguous K4 dword per output lane.
            for (; kk + 3 < max_kk; kk += 4)
            {
#if !__AVXVNNIINT8__
                __m128i _p = _mm_loadu_si32(p0);
                _p = _mm_add_epi8(_p, _mm_set1_epi8(127));
                _mm_storeu_si32(pp, _p);
#else  // __AVXVNNIINT8__
                _mm_storeu_si32(pp, _mm_loadu_si32(p0));
#endif // __AVXVNNIINT8__
                pp += 4;
                p0 += 4;
            }
#else
            // AVX2/SSE2 consumes two K2 vectors for each real K4 region.
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __SSE2__
                _mm_storeu_si32(pp, _mm_loadu_si32(p0));
#else
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
#endif // __SSE2__
                pp += 4;
                p0 += 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__ \
// K2/K1 are always signed and compact, including classic VNNI.
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __SSE2__
                _mm_storeu_si16(pp, _mm_loadu_si16(p0));
#else
                pp[0] = p0[0];
                pp[1] = p0[1];
#endif // __SSE2__
                pp += 2;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                *pp++ = (unsigned char)*p0++;
            }

            pd[0] = 1.f / *ps0++;
            pd += 1;
        }
    }
}

static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        quantize_A_tile_wq_int8_avx512vnni(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        quantize_A_tile_wq_int8_avxvnniint8(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        quantize_A_tile_wq_int8_avxvnni(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        quantize_A_tile_wq_int8_avx2(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

    signed char* outptr = AT_tile;
    const int out_hstep = AT_tile.w;
    float* descale_ptr = AT_descales_tile;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const int block_count = (K + block_size - 1) / block_size;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32((int)A_hstep));
    for (; ii + 15 < max_ii; ii += 16)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

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
            int kk_absmax = 0;
            for (; kk_absmax + 15 < max_kk; kk_absmax += 16)
            {
                __m512 _s;
                if (input_scale_ptr)
                    _s = _mm512_loadu_ps(input_scale_ptr + k0 + kk_absmax);

                __m512 _p = _mm512_loadu_ps(p0 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax0 = _mm512_max_ps(_absmax0, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax1 = _mm512_max_ps(_absmax1, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 2 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax2 = _mm512_max_ps(_absmax2, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 3 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax3 = _mm512_max_ps(_absmax3, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 4 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax4 = _mm512_max_ps(_absmax4, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 5 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax5 = _mm512_max_ps(_absmax5, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 6 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax6 = _mm512_max_ps(_absmax6, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 7 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax7 = _mm512_max_ps(_absmax7, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 8 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax8 = _mm512_max_ps(_absmax8, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 9 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmax9 = _mm512_max_ps(_absmax9, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 10 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmaxa = _mm512_max_ps(_absmaxa, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 11 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmaxb = _mm512_max_ps(_absmaxb, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 12 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmaxc = _mm512_max_ps(_absmaxc, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 13 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmaxd = _mm512_max_ps(_absmaxd, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 14 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmaxe = _mm512_max_ps(_absmaxe, abs512_ps(_p));
                _p = _mm512_loadu_ps(p0 + A_hstep * 15 + kk_absmax);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _s);
                _absmaxf = _mm512_max_ps(_absmaxf, abs512_ps(_p));
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
            for (; kk_absmax + 3 < max_kk; kk_absmax += 4)
            {
                __m128 _s = input_scale_ptr ? _mm_loadu_ps(input_scale_ptr + k0 + kk_absmax) : _mm_set1_ps(1.f);
                absmax0 = std::max(absmax0, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + kk_absmax), _s))));
                absmax1 = std::max(absmax1, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep + kk_absmax), _s))));
                absmax2 = std::max(absmax2, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 2 + kk_absmax), _s))));
                absmax3 = std::max(absmax3, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 3 + kk_absmax), _s))));
                absmax4 = std::max(absmax4, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 4 + kk_absmax), _s))));
                absmax5 = std::max(absmax5, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 5 + kk_absmax), _s))));
                absmax6 = std::max(absmax6, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 6 + kk_absmax), _s))));
                absmax7 = std::max(absmax7, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 7 + kk_absmax), _s))));
                absmax8 = std::max(absmax8, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 8 + kk_absmax), _s))));
                absmax9 = std::max(absmax9, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 9 + kk_absmax), _s))));
                absmaxa = std::max(absmaxa, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 10 + kk_absmax), _s))));
                absmaxb = std::max(absmaxb, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 11 + kk_absmax), _s))));
                absmaxc = std::max(absmaxc, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 12 + kk_absmax), _s))));
                absmaxd = std::max(absmaxd, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 13 + kk_absmax), _s))));
                absmaxe = std::max(absmaxe, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 14 + kk_absmax), _s))));
                absmaxf = std::max(absmaxf, _mm_reduce_max_ps(abs_ps(_mm_mul_ps(_mm_loadu_ps(p0 + A_hstep * 15 + kk_absmax), _s))));
            }
            for (; kk_absmax < max_kk; kk_absmax++)
            {
                const float s = input_scale_ptr ? input_scale_ptr[k0 + kk_absmax] : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0[kk_absmax] * s));
                absmax1 = std::max(absmax1, fabsf(p0[A_hstep + kk_absmax] * s));
                absmax2 = std::max(absmax2, fabsf(p0[A_hstep * 2 + kk_absmax] * s));
                absmax3 = std::max(absmax3, fabsf(p0[A_hstep * 3 + kk_absmax] * s));
                absmax4 = std::max(absmax4, fabsf(p0[A_hstep * 4 + kk_absmax] * s));
                absmax5 = std::max(absmax5, fabsf(p0[A_hstep * 5 + kk_absmax] * s));
                absmax6 = std::max(absmax6, fabsf(p0[A_hstep * 6 + kk_absmax] * s));
                absmax7 = std::max(absmax7, fabsf(p0[A_hstep * 7 + kk_absmax] * s));
                absmax8 = std::max(absmax8, fabsf(p0[A_hstep * 8 + kk_absmax] * s));
                absmax9 = std::max(absmax9, fabsf(p0[A_hstep * 9 + kk_absmax] * s));
                absmaxa = std::max(absmaxa, fabsf(p0[A_hstep * 10 + kk_absmax] * s));
                absmaxb = std::max(absmaxb, fabsf(p0[A_hstep * 11 + kk_absmax] * s));
                absmaxc = std::max(absmaxc, fabsf(p0[A_hstep * 12 + kk_absmax] * s));
                absmaxd = std::max(absmaxd, fabsf(p0[A_hstep * 13 + kk_absmax] * s));
                absmaxe = std::max(absmaxe, fabsf(p0[A_hstep * 14 + kk_absmax] * s));
                absmaxf = std::max(absmaxf, fabsf(p0[A_hstep * 15 + kk_absmax] * s));
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
            signed char* pp = pp0;
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                {
                    __m256 _p0 = _mm256_loadu_ps(p0 + kk);
                    __m256 _p1 = _mm256_loadu_ps(p0 + A_hstep + kk);
                    __m256 _p2 = _mm256_loadu_ps(p0 + A_hstep * 2 + kk);
                    __m256 _p3 = _mm256_loadu_ps(p0 + A_hstep * 3 + kk);
                    __m256 _p4 = _mm256_loadu_ps(p0 + A_hstep * 4 + kk);
                    __m256 _p5 = _mm256_loadu_ps(p0 + A_hstep * 5 + kk);
                    __m256 _p6 = _mm256_loadu_ps(p0 + A_hstep * 6 + kk);
                    __m256 _p7 = _mm256_loadu_ps(p0 + A_hstep * 7 + kk);
                    __m256 _p8 = _mm256_loadu_ps(p0 + A_hstep * 8 + kk);
                    __m256 _p9 = _mm256_loadu_ps(p0 + A_hstep * 9 + kk);
                    __m256 _pa = _mm256_loadu_ps(p0 + A_hstep * 10 + kk);
                    __m256 _pb = _mm256_loadu_ps(p0 + A_hstep * 11 + kk);
                    __m256 _pc = _mm256_loadu_ps(p0 + A_hstep * 12 + kk);
                    __m256 _pd = _mm256_loadu_ps(p0 + A_hstep * 13 + kk);
                    __m256 _pe = _mm256_loadu_ps(p0 + A_hstep * 14 + kk);
                    __m256 _pf = _mm256_loadu_ps(p0 + A_hstep * 15 + kk);
                    if (input_scale_ptr)
                    {
                        __m256 _s = _mm256_loadu_ps(input_scale_ptr + k0 + kk);
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
                    }
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
                    __m256 _p0 = _mm256_loadu_ps(p0 + kk + 8);
                    __m256 _p1 = _mm256_loadu_ps(p0 + A_hstep + kk + 8);
                    __m256 _p2 = _mm256_loadu_ps(p0 + A_hstep * 2 + kk + 8);
                    __m256 _p3 = _mm256_loadu_ps(p0 + A_hstep * 3 + kk + 8);
                    __m256 _p4 = _mm256_loadu_ps(p0 + A_hstep * 4 + kk + 8);
                    __m256 _p5 = _mm256_loadu_ps(p0 + A_hstep * 5 + kk + 8);
                    __m256 _p6 = _mm256_loadu_ps(p0 + A_hstep * 6 + kk + 8);
                    __m256 _p7 = _mm256_loadu_ps(p0 + A_hstep * 7 + kk + 8);
                    __m256 _p8 = _mm256_loadu_ps(p0 + A_hstep * 8 + kk + 8);
                    __m256 _p9 = _mm256_loadu_ps(p0 + A_hstep * 9 + kk + 8);
                    __m256 _pa = _mm256_loadu_ps(p0 + A_hstep * 10 + kk + 8);
                    __m256 _pb = _mm256_loadu_ps(p0 + A_hstep * 11 + kk + 8);
                    __m256 _pc = _mm256_loadu_ps(p0 + A_hstep * 12 + kk + 8);
                    __m256 _pd = _mm256_loadu_ps(p0 + A_hstep * 13 + kk + 8);
                    __m256 _pe = _mm256_loadu_ps(p0 + A_hstep * 14 + kk + 8);
                    __m256 _pf = _mm256_loadu_ps(p0 + A_hstep * 15 + kk + 8);
                    if (input_scale_ptr)
                    {
                        __m256 _s = _mm256_loadu_ps(input_scale_ptr + k0 + kk + 8);
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
                    }
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
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0 + kk);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep + kk);
                __m128 _p2 = _mm_loadu_ps(p0 + A_hstep * 2 + kk);
                __m128 _p3 = _mm_loadu_ps(p0 + A_hstep * 3 + kk);
                __m128 _p4 = _mm_loadu_ps(p0 + A_hstep * 4 + kk);
                __m128 _p5 = _mm_loadu_ps(p0 + A_hstep * 5 + kk);
                __m128 _p6 = _mm_loadu_ps(p0 + A_hstep * 6 + kk);
                __m128 _p7 = _mm_loadu_ps(p0 + A_hstep * 7 + kk);
                __m128 _p8 = _mm_loadu_ps(p0 + A_hstep * 8 + kk);
                __m128 _p9 = _mm_loadu_ps(p0 + A_hstep * 9 + kk);
                __m128 _pa = _mm_loadu_ps(p0 + A_hstep * 10 + kk);
                __m128 _pb = _mm_loadu_ps(p0 + A_hstep * 11 + kk);
                __m128 _pc = _mm_loadu_ps(p0 + A_hstep * 12 + kk);
                __m128 _pd = _mm_loadu_ps(p0 + A_hstep * 13 + kk);
                __m128 _pe = _mm_loadu_ps(p0 + A_hstep * 14 + kk);
                __m128 _pf = _mm_loadu_ps(p0 + A_hstep * 15 + kk);
                if (input_scale_ptr)
                {
                    __m128 _s = _mm_loadu_ps(input_scale_ptr + k0 + kk);
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
                }
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
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                _mm512_storeu_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_i32gather_ps(_vindex, p0 + kk, sizeof(float));
                __m512 _p1 = _mm512_i32gather_ps(_vindex, p0 + kk + 1, sizeof(float));
                if (input_scale_ptr)
                {
                    _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(input_scale_ptr[k0 + kk + 1]));
                }
                __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                pp += 32;
            }
            if (kk < max_kk)
            {
                __m512 _p = _mm512_i32gather_ps(_vindex, p0 + kk, sizeof(float));
                if (input_scale_ptr)
                {
                    _p = _mm512_mul_ps(_p, _mm512_set1_ps(input_scale_ptr[k0 + kk]));
                }
                _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
            }

            p0 += max_kk;
            pd += 16;
#if __AVX512VNNI__
            pp0 += (max_kk + (max_kk >= 4 ? 4 : 0)) * 16;
#else
            pp0 += max_kk * 16;
#endif
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            __m256 _absmax0 = _mm256_setzero_ps();
            __m256 _absmax1 = _mm256_setzero_ps();
            __m256 _absmax2 = _mm256_setzero_ps();
            __m256 _absmax3 = _mm256_setzero_ps();
            __m256 _absmax4 = _mm256_setzero_ps();
            __m256 _absmax5 = _mm256_setzero_ps();
            __m256 _absmax6 = _mm256_setzero_ps();
            __m256 _absmax7 = _mm256_setzero_ps();
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _s;
                if (input_scale_ptr)
                    _s = _mm256_loadu_ps(input_scale_ptr + k0 + kk);

                __m256 _p = _mm256_loadu_ps(p0 + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _s);
                _absmax0 = _mm256_max_ps(_absmax0, abs256_ps(_p));
                _p = _mm256_loadu_ps(p0 + A_hstep + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _s);
                _absmax1 = _mm256_max_ps(_absmax1, abs256_ps(_p));
                _p = _mm256_loadu_ps(p0 + A_hstep * 2 + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _s);
                _absmax2 = _mm256_max_ps(_absmax2, abs256_ps(_p));
                _p = _mm256_loadu_ps(p0 + A_hstep * 3 + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _s);
                _absmax3 = _mm256_max_ps(_absmax3, abs256_ps(_p));
                _p = _mm256_loadu_ps(p0 + A_hstep * 4 + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _s);
                _absmax4 = _mm256_max_ps(_absmax4, abs256_ps(_p));
                _p = _mm256_loadu_ps(p0 + A_hstep * 5 + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _s);
                _absmax5 = _mm256_max_ps(_absmax5, abs256_ps(_p));
                _p = _mm256_loadu_ps(p0 + A_hstep * 6 + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _s);
                _absmax6 = _mm256_max_ps(_absmax6, abs256_ps(_p));
                _p = _mm256_loadu_ps(p0 + A_hstep * 7 + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _s);
                _absmax7 = _mm256_max_ps(_absmax7, abs256_ps(_p));
            }

            float absmax0 = _mm256_reduce_max_ps(_absmax0);
            float absmax1 = _mm256_reduce_max_ps(_absmax1);
            float absmax2 = _mm256_reduce_max_ps(_absmax2);
            float absmax3 = _mm256_reduce_max_ps(_absmax3);
            float absmax4 = _mm256_reduce_max_ps(_absmax4);
            float absmax5 = _mm256_reduce_max_ps(_absmax5);
            float absmax6 = _mm256_reduce_max_ps(_absmax6);
            float absmax7 = _mm256_reduce_max_ps(_absmax7);
            for (; kk < max_kk; kk++)
            {
                const float s = input_scale_ptr ? input_scale_ptr[k0 + kk] : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0[kk] * s));
                absmax1 = std::max(absmax1, fabsf(p0[A_hstep + kk] * s));
                absmax2 = std::max(absmax2, fabsf(p0[A_hstep * 2 + kk] * s));
                absmax3 = std::max(absmax3, fabsf(p0[A_hstep * 3 + kk] * s));
                absmax4 = std::max(absmax4, fabsf(p0[A_hstep * 4 + kk] * s));
                absmax5 = std::max(absmax5, fabsf(p0[A_hstep * 5 + kk] * s));
                absmax6 = std::max(absmax6, fabsf(p0[A_hstep * 6 + kk] * s));
                absmax7 = std::max(absmax7, fabsf(p0[A_hstep * 7 + kk] * s));
            }

            __m256 _absmax = _mm256_setr_ps(absmax0, absmax1, absmax2, absmax3, absmax4, absmax5, absmax6, absmax7);
            __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
            __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
            __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
            _mm256_storeu_ps(pd, _descale);

            signed char* pp = pp0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            __m256i _w_shift = _mm256_setzero_si256();
#endif
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0 + kk);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep + kk);
                __m128 _p2 = _mm_loadu_ps(p0 + A_hstep * 2 + kk);
                __m128 _p3 = _mm_loadu_ps(p0 + A_hstep * 3 + kk);
                __m128 _p4 = _mm_loadu_ps(p0 + A_hstep * 4 + kk);
                __m128 _p5 = _mm_loadu_ps(p0 + A_hstep * 5 + kk);
                __m128 _p6 = _mm_loadu_ps(p0 + A_hstep * 6 + kk);
                __m128 _p7 = _mm_loadu_ps(p0 + A_hstep * 7 + kk);
                if (input_scale_ptr)
                {
                    __m128 _s = _mm_loadu_ps(input_scale_ptr + k0 + kk);
                    _p0 = _mm_mul_ps(_p0, _s);
                    _p1 = _mm_mul_ps(_p1, _s);
                    _p2 = _mm_mul_ps(_p2, _s);
                    _p3 = _mm_mul_ps(_p3, _s);
                    _p4 = _mm_mul_ps(_p4, _s);
                    _p5 = _mm_mul_ps(_p5, _s);
                    _p6 = _mm_mul_ps(_p6, _s);
                    _p7 = _mm_mul_ps(_p7, _s);
                }

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
#else
                _mm_storeu_si128((__m128i*)pp, _q01);
                _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 32;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _mm256_storeu_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#endif
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32((int)A_hstep));
                __m256 _p0 = _mm256_i32gather_ps(p0 + kk, _vindex, sizeof(float));
                __m256 _p1 = _mm256_i32gather_ps(p0 + kk + 1, _vindex, sizeof(float));
                if (input_scale_ptr)
                {
                    _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(input_scale_ptr[k0 + kk + 1]));
                }
                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                __m128i _q = float2int8_avx(_p0, _p1);
                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _q = _mm_shuffle_epi8(_q, _si);
                _mm_storeu_si128((__m128i*)pp, _q);
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32((int)A_hstep));
                __m256 _p = _mm256_i32gather_ps(p0 + kk, _vindex, sizeof(float));
                if (input_scale_ptr)
                {
                    _p = _mm256_mul_ps(_p, _mm256_set1_ps(input_scale_ptr[k0 + kk]));
                }
                _mm_storeu_si64(pp, _mm_cvtsi64_si128(float2int8_avx(_mm256_mul_ps(_p, _scale))));
                pp += 8;
            }

            p0 += max_kk;
            pd += 8;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            pp0 += (max_kk + (max_kk >= 4 ? 4 : 0)) * 8;
#else
            pp0 += max_kk * 8;
#endif
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            __m128 _absmax0 = _mm_setzero_ps();
            __m128 _absmax1 = _mm_setzero_ps();
            __m128 _absmax2 = _mm_setzero_ps();
            __m128 _absmax3 = _mm_setzero_ps();
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0 + kk);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep + kk);
                __m128 _p2 = _mm_loadu_ps(p0 + A_hstep * 2 + kk);
                __m128 _p3 = _mm_loadu_ps(p0 + A_hstep * 3 + kk);
                if (input_scale_ptr)
                {
                    __m128 _s = _mm_loadu_ps(input_scale_ptr + k0 + kk);
                    _p0 = _mm_mul_ps(_p0, _s);
                    _p1 = _mm_mul_ps(_p1, _s);
                    _p2 = _mm_mul_ps(_p2, _s);
                    _p3 = _mm_mul_ps(_p3, _s);
                }
                _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p0));
                _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
                _absmax2 = _mm_max_ps(_absmax2, abs_ps(_p2));
                _absmax3 = _mm_max_ps(_absmax3, abs_ps(_p3));
            }

            float absmax0 = _mm_reduce_max_ps(_absmax0);
            float absmax1 = _mm_reduce_max_ps(_absmax1);
            float absmax2 = _mm_reduce_max_ps(_absmax2);
            float absmax3 = _mm_reduce_max_ps(_absmax3);
            for (; kk < max_kk; kk++)
            {
                const float s = input_scale_ptr ? input_scale_ptr[k0 + kk] : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0[kk] * s));
                absmax1 = std::max(absmax1, fabsf(p0[A_hstep + kk] * s));
                absmax2 = std::max(absmax2, fabsf(p0[A_hstep * 2 + kk] * s));
                absmax3 = std::max(absmax3, fabsf(p0[A_hstep * 3 + kk] * s));
            }

            __m128 _absmax = _mm_setr_ps(absmax0, absmax1, absmax2, absmax3);
            __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
            __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
            __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
            _mm_storeu_ps(pd, _descale);

            signed char* pp = pp0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            __m128i _w_shift = _mm_setzero_si128();
#endif
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0 + kk);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep + kk);
                __m128 _p2 = _mm_loadu_ps(p0 + A_hstep * 2 + kk);
                __m128 _p3 = _mm_loadu_ps(p0 + A_hstep * 3 + kk);
                if (input_scale_ptr)
                {
                    __m128 _s = _mm_loadu_ps(input_scale_ptr + k0 + kk);
                    _p0 = _mm_mul_ps(_p0, _s);
                    _p1 = _mm_mul_ps(_p1, _s);
                    _p2 = _mm_mul_ps(_p2, _s);
                    _p3 = _mm_mul_ps(_p3, _s);
                }
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
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _mm_storeu_si128((__m128i*)pp, _w_shift);
                pp += 16;
            }
#endif
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_setr_ps(p0[kk], p0[A_hstep + kk], p0[A_hstep * 2 + kk], p0[A_hstep * 3 + kk]);
                __m128 _p1 = _mm_setr_ps(p0[kk + 1], p0[A_hstep + kk + 1], p0[A_hstep * 2 + kk + 1], p0[A_hstep * 3 + kk + 1]);
                if (input_scale_ptr)
                {
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(input_scale_ptr[k0 + kk + 1]));
                }
                __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p = _mm_setr_ps(p0[kk], p0[A_hstep + kk], p0[A_hstep * 2 + kk], p0[A_hstep * 3 + kk]);
                if (input_scale_ptr)
                {
                    _p = _mm_mul_ps(_p, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                }
                _mm_storeu_si32(pp, _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p, _scale))));
                pp += 4;
            }

            p0 += max_kk;
            pd += 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            pp0 += (max_kk + (max_kk >= 4 ? 4 : 0)) * 4;
#else
            pp0 += max_kk * 4;
#endif
        }
    }
#endif // __SSE2__
#if __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            __m128 _absmax0 = _mm_setzero_ps();
            __m128 _absmax1 = _mm_setzero_ps();
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0 + kk);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep + kk);
                if (input_scale_ptr)
                {
                    __m128 _s = _mm_loadu_ps(input_scale_ptr + k0 + kk);
                    _p0 = _mm_mul_ps(_p0, _s);
                    _p1 = _mm_mul_ps(_p1, _s);
                }
                _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p0));
                _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
            }

            float absmax0 = _mm_reduce_max_ps(_absmax0);
            float absmax1 = _mm_reduce_max_ps(_absmax1);
            for (; kk < max_kk; kk++)
            {
                const float s = input_scale_ptr ? input_scale_ptr[k0 + kk] : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0[kk] * s));
                absmax1 = std::max(absmax1, fabsf(p0[A_hstep + kk] * s));
            }

            __m128 _absmax = _mm_setr_ps(absmax0, absmax1, 0.f, 0.f);
            __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
            __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
            __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
            _mm_storel_pi((__m64*)pd, _descale);

            signed char* pp = pp0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            __m128i _w_shift = _mm_setzero_si128();
#endif
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0 + kk);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep + kk);
                if (input_scale_ptr)
                {
                    __m128 _s = _mm_loadu_ps(input_scale_ptr + k0 + kk);
                    _p0 = _mm_mul_ps(_p0, _s);
                    _p1 = _mm_mul_ps(_p1, _s);
                }
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
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _mm_storel_epi64((__m128i*)pp, _w_shift);
                pp += 8;
            }
#endif
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_setr_ps(p0[kk], p0[A_hstep + kk], 0.f, 0.f);
                __m128 _p1 = _mm_setr_ps(p0[kk + 1], p0[A_hstep + kk + 1], 0.f, 0.f);
                if (input_scale_ptr)
                {
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(input_scale_ptr[k0 + kk + 1]));
                }
                __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                _mm_storeu_si32(pp, _mm_unpacklo_epi8(_q0, _q1));
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p = _mm_setr_ps(p0[kk], p0[A_hstep + kk], 0.f, 0.f);
                if (input_scale_ptr)
                {
                    _p = _mm_mul_ps(_p, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                }
                _mm_storeu_si16(pp, _mm_cvtsi32_si128((unsigned short)float2int8_sse(_mm_mul_ps(_p, _scale))));
                pp += 2;
            }

            p0 += max_kk;
            pd += 2;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            pp0 += (max_kk + (max_kk >= 4 ? 4 : 0)) * 2;
#else
            pp0 += max_kk * 2;
#endif
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;
        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = p0[kk];
                float v1 = p0[A_hstep + kk];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k0 + kk];
                    v0 *= s;
                    v1 *= s;
                }
                absmax0 = std::max(absmax0, (float)fabsf(v0));
                absmax1 = std::max(absmax1, (float)fabsf(v1));
            }

            float scale0 = 0.f;
            float scale1 = 0.f;
            if (absmax0 != 0.f)
            {
                scale0 = 127.f / absmax0;
            }
            if (absmax1 != 0.f)
            {
                scale1 = 127.f / absmax1;
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;

            signed char* pp = pp0;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = p0[kk];
                float v1 = p0[A_hstep + kk];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k0 + kk];
                    v0 *= s;
                    v1 *= s;
                }
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
            }

            p0 += max_kk;
            pp0 += max_kk * 2;
            pd += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            signed char* pp = pp0;
            float absmax = 0.f;

            int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _absmax512 = _mm512_setzero_ps();
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p = _mm512_loadu_ps(p0 + kk);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _mm512_loadu_ps(input_scale_ptr + k0 + kk));
                _absmax512 = _mm512_max_ps(_absmax512, abs512_ps(_p));
            }
            absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax512));
#endif // __AVX512F__
            __m256 _absmax256 = _mm256_setzero_ps();
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p = _mm256_loadu_ps(p0 + kk);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _mm256_loadu_ps(input_scale_ptr + k0 + kk));
                _absmax256 = _mm256_max_ps(_absmax256, abs256_ps(_p));
            }
            absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax256));
#endif // __AVX__
            __m128 _absmax128 = _mm_setzero_ps();
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p = _mm_loadu_ps(p0 + kk);
                if (input_scale_ptr)
                    _p = _mm_mul_ps(_p, _mm_loadu_ps(input_scale_ptr + k0 + kk));
                _absmax128 = _mm_max_ps(_absmax128, abs_ps(_p));
            }
            absmax = std::max(absmax, _mm_reduce_max_ps(_absmax128));
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                float v = p0[kk];
                if (input_scale_ptr)
                    v *= input_scale_ptr[k0 + kk];
                absmax = std::max(absmax, (float)fabsf(v));
            }

            if (absmax == 0.f)
            {
                pd[0] = 0.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                memset(pp, 0, max_kk >= 4 ? max_kk + 4 : max_kk);
                pp0 += max_kk + (max_kk >= 4 ? 4 : 0);
#else
                memset(pp, 0, max_kk);
                pp0 += max_kk;
#endif
                p0 += max_kk;
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
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p = _mm512_loadu_ps(p0 + kk);
                if (input_scale_ptr)
                {
                    _p = _mm512_mul_ps(_p, _mm512_loadu_ps(input_scale_ptr + k0 + kk));
                }
                __m128i _q = float2int8_avx512(_mm512_mul_ps(_p, _scale512));
                _mm_storeu_si128((__m128i*)pp, _q);
                pp += 16;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _q16 = _mm256_cvtepi8_epi16(_q);
                __m256i _q32 = _mm256_madd_epi16(_q16, _mm256_set1_epi16(1));
                w_shift += _mm_reduce_add_epi32(_mm256_castsi256_si128(_q32));
                w_shift += _mm_reduce_add_epi32(_mm256_extracti128_si256(_q32, 1));
#endif
            }
#endif // __AVX512F__
            __m256 _scale256 = _mm256_set1_ps(scale);
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p = _mm256_loadu_ps(p0 + kk);
                if (input_scale_ptr)
                {
                    _p = _mm256_mul_ps(_p, _mm256_loadu_ps(input_scale_ptr + k0 + kk));
                }
                const int64_t q = float2int8_avx(_mm256_mul_ps(_p, _scale256));
                _mm_storeu_si64(pp, _mm_cvtsi64_si128(q));
                pp += 8;
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
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p = _mm_loadu_ps(p0 + kk);
                if (input_scale_ptr)
                {
                    _p = _mm_mul_ps(_p, _mm_loadu_ps(input_scale_ptr + k0 + kk));
                }
                const int32_t q = float2int8_sse(_mm_mul_ps(_p, _scale128));
                _mm_storeu_si32(pp, _mm_cvtsi32_si128(q));
                pp += 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _q8 = _mm_cvtsi32_si128(q);
                __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
            }
#endif // __SSE2__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _mm_storeu_si32(pp, _mm_cvtsi32_si128(w_shift * 127));
                pp += 4;
            }
#endif
            for (; kk < max_kk; kk++)
            {
                float v = p0[kk];
                if (input_scale_ptr)
                {
                    v *= input_scale_ptr[k0 + kk];
                }
                *pp++ = float2int8(v * scale);
            }

            p0 += max_kk;
            pd++;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            pp0 += max_kk + (max_kk >= 4 ? 4 : 0);
#else
            pp0 += max_kk;
#endif
        }
    }
}

static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_quantize_A_tile_wq_int8_avx512vnni(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        transpose_quantize_A_tile_wq_int8_avxvnniint8(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_quantize_A_tile_wq_int8_avxvnni(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_quantize_A_tile_wq_int8_avx2(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

    signed char* outptr = AT_tile;
    const int out_hstep = AT_tile.w;
    float* descale_ptr = AT_descales_tile;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const int block_count = (K + block_size - 1) / block_size;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const float* p0 = (const float*)A + i + ii;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            __m512 _absmax = _mm512_setzero_ps();
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m512 _p = _mm512_loadu_ps(p0 + kk * A_hstep);
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _mm512_set1_ps(input_scale_ptr[k0 + kk]));
                _absmax = _mm512_max_ps(_absmax, abs512_ps(_p));
            }

            __m512 _descale = _mm512_div_ps(_absmax, _mm512_set1_ps(127.f));
            __mmask16 _nonzero = _mm512_cmp_ps_mask(_absmax, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            __m512 _scale = _mm512_maskz_div_ps(_nonzero, _mm512_set1_ps(127.f), _absmax);
            _mm512_storeu_ps(pd, _descale);

#if __AVX512VNNI__
            __m512i _w_shift = _mm512_setzero_si512();
            __m512i _v127 = _mm512_set1_epi8(127);
#endif
            signed char* pp = pp0;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0 + kk * A_hstep);
                __m512 _p1 = _mm512_loadu_ps(p0 + (kk + 1) * A_hstep);
                __m512 _p2 = _mm512_loadu_ps(p0 + (kk + 2) * A_hstep);
                __m512 _p3 = _mm512_loadu_ps(p0 + (kk + 3) * A_hstep);
                if (input_scale_ptr)
                {
                    _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(input_scale_ptr[k0 + kk + 1]));
                    _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(input_scale_ptr[k0 + kk + 2]));
                    _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(input_scale_ptr[k0 + kk + 3]));
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
            }
#endif // __AVX512VNNI__
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                _mm512_storeu_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_loadu_ps(p0 + kk * A_hstep);
                __m512 _p1 = _mm512_loadu_ps(p0 + (kk + 1) * A_hstep);
                if (input_scale_ptr)
                {
                    _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(input_scale_ptr[k0 + kk + 1]));
                }
                __m128i _q0 = float2int8_avx512(_mm512_mul_ps(_p0, _scale));
                __m128i _q1 = float2int8_avx512(_mm512_mul_ps(_p1, _scale));
                _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpackhi_epi8(_q0, _q1));
                pp += 32;
            }
            if (kk < max_kk)
            {
                __m512 _p = _mm512_loadu_ps(p0 + kk * A_hstep);
                if (input_scale_ptr)
                {
                    _p = _mm512_mul_ps(_p, _mm512_set1_ps(input_scale_ptr[k0 + kk]));
                }
                _mm_storeu_si128((__m128i*)pp, float2int8_avx512(_mm512_mul_ps(_p, _scale)));
            }

            p0 += max_kk * A_hstep;
            pd += 16;
#if __AVX512VNNI__
            pp0 += (max_kk + (max_kk >= 4 ? 4 : 0)) * 16;
#else
            pp0 += max_kk * 16;
#endif
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + i + ii;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            __m256 _absmax = _mm256_setzero_ps();
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _p = _mm256_loadu_ps(p0 + kk * A_hstep);
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _mm256_set1_ps(input_scale_ptr[k0 + kk]));
                _absmax = _mm256_max_ps(_absmax, abs256_ps(_p));
            }

            __m256 _descale = _mm256_div_ps(_absmax, _mm256_set1_ps(127.f));
            __m256 _nonzero = _mm256_cmp_ps(_absmax, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _absmax_nonzero = _mm256_blendv_ps(_mm256_set1_ps(1.f), _absmax, _nonzero);
            __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(127.f), _absmax_nonzero), _nonzero);
            _mm256_storeu_ps(pd, _descale);

            signed char* pp = pp0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            __m256i _w_shift = _mm256_setzero_si256();
#endif
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_loadu_ps(p0 + kk * A_hstep);
                __m256 _p1 = _mm256_loadu_ps(p0 + (kk + 1) * A_hstep);
                __m256 _p2 = _mm256_loadu_ps(p0 + (kk + 2) * A_hstep);
                __m256 _p3 = _mm256_loadu_ps(p0 + (kk + 3) * A_hstep);
                if (input_scale_ptr)
                {
                    _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(input_scale_ptr[k0 + kk + 1]));
                    _p2 = _mm256_mul_ps(_p2, _mm256_set1_ps(input_scale_ptr[k0 + kk + 2]));
                    _p3 = _mm256_mul_ps(_p3, _mm256_set1_ps(input_scale_ptr[k0 + kk + 3]));
                }
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
#else
                _mm_storeu_si128((__m128i*)pp, _q01);
                _mm_storeu_si128((__m128i*)(pp + 16), _q23);
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 32;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _mm256_storeu_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#endif
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _p0 = _mm256_loadu_ps(p0 + kk * A_hstep);
                __m256 _p1 = _mm256_loadu_ps(p0 + (kk + 1) * A_hstep);
                if (input_scale_ptr)
                {
                    _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(input_scale_ptr[k0 + kk + 1]));
                }
                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                __m128i _q = float2int8_avx(_p0, _p1);
                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _q = _mm_shuffle_epi8(_q, _si);
                _mm_storeu_si128((__m128i*)pp, _q);
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                __m256 _p = _mm256_loadu_ps(p0 + kk * A_hstep);
                if (input_scale_ptr)
                {
                    _p = _mm256_mul_ps(_p, _mm256_set1_ps(input_scale_ptr[k0 + kk]));
                }
                _mm_storeu_si64(pp, _mm_cvtsi64_si128(float2int8_avx(_mm256_mul_ps(_p, _scale))));
                pp += 8;
            }

            p0 += max_kk * A_hstep;
            pd += 8;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            pp0 += (max_kk + (max_kk >= 4 ? 4 : 0)) * 8;
#else
            pp0 += max_kk * 8;
#endif
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + i + ii;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            __m128 _absmax = _mm_setzero_ps();
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _p = _mm_loadu_ps(p0 + kk * A_hstep);
                if (input_scale_ptr)
                    _p = _mm_mul_ps(_p, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                _absmax = _mm_max_ps(_absmax, abs_ps(_p));
            }

            __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
            __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
            __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
            _mm_storeu_ps(pd, _descale);

            signed char* pp = pp0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            __m128i _w_shift = _mm_setzero_si128();
#endif
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0 + kk * A_hstep);
                __m128 _p1 = _mm_loadu_ps(p0 + (kk + 1) * A_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + (kk + 2) * A_hstep);
                __m128 _p3 = _mm_loadu_ps(p0 + (kk + 3) * A_hstep);
                if (input_scale_ptr)
                {
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(input_scale_ptr[k0 + kk + 1]));
                    _p2 = _mm_mul_ps(_p2, _mm_set1_ps(input_scale_ptr[k0 + kk + 2]));
                    _p3 = _mm_mul_ps(_p3, _mm_set1_ps(input_scale_ptr[k0 + kk + 3]));
                }
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
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _mm_storeu_si128((__m128i*)pp, _w_shift);
                pp += 16;
            }
#endif
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_loadu_ps(p0 + kk * A_hstep);
                __m128 _p1 = _mm_loadu_ps(p0 + (kk + 1) * A_hstep);
                if (input_scale_ptr)
                {
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(input_scale_ptr[k0 + kk + 1]));
                }
                __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                _mm_storel_epi64((__m128i*)pp, _mm_unpacklo_epi8(_q0, _q1));
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p = _mm_loadu_ps(p0 + kk * A_hstep);
                if (input_scale_ptr)
                {
                    _p = _mm_mul_ps(_p, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                }
                _mm_storeu_si32(pp, _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p, _scale))));
                pp += 4;
            }

            p0 += max_kk * A_hstep;
            pd += 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            pp0 += (max_kk + (max_kk >= 4 ? 4 : 0)) * 4;
#else
            pp0 += max_kk * 4;
#endif
        }
    }
#endif // __SSE2__
#if __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + i + ii;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            __m128 _absmax = _mm_setzero_ps();
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _p = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(p0 + kk * A_hstep));
                if (input_scale_ptr)
                    _p = _mm_mul_ps(_p, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                _absmax = _mm_max_ps(_absmax, abs_ps(_p));
            }

            __m128 _descale = _mm_div_ps(_absmax, _mm_set1_ps(127.f));
            __m128 _nonzero = _mm_cmpneq_ps(_absmax, _mm_setzero_ps());
            __m128 _absmax_nonzero = _mm_or_ps(_mm_and_ps(_absmax, _nonzero), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(127.f), _absmax_nonzero), _nonzero);
            _mm_storel_pi((__m64*)pd, _descale);

            signed char* pp = pp0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            __m128i _w_shift = _mm_setzero_si128();
#endif
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(p0 + kk * A_hstep));
                __m128 _p1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(p0 + (kk + 1) * A_hstep));
                __m128 _p2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(p0 + (kk + 2) * A_hstep));
                __m128 _p3 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(p0 + (kk + 3) * A_hstep));
                if (input_scale_ptr)
                {
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(input_scale_ptr[k0 + kk + 1]));
                    _p2 = _mm_mul_ps(_p2, _mm_set1_ps(input_scale_ptr[k0 + kk + 2]));
                    _p3 = _mm_mul_ps(_p3, _mm_set1_ps(input_scale_ptr[k0 + kk + 3]));
                }
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
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _mm_storel_epi64((__m128i*)pp, _w_shift);
                pp += 8;
            }
#endif
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(p0 + kk * A_hstep));
                __m128 _p1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(p0 + (kk + 1) * A_hstep));
                if (input_scale_ptr)
                {
                    _p0 = _mm_mul_ps(_p0, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                    _p1 = _mm_mul_ps(_p1, _mm_set1_ps(input_scale_ptr[k0 + kk + 1]));
                }
                __m128i _q0 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p0, _scale)));
                __m128i _q1 = _mm_cvtsi32_si128(float2int8_sse(_mm_mul_ps(_p1, _scale)));
                _mm_storeu_si32(pp, _mm_unpacklo_epi8(_q0, _q1));
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(p0 + kk * A_hstep));
                if (input_scale_ptr)
                {
                    _p = _mm_mul_ps(_p, _mm_set1_ps(input_scale_ptr[k0 + kk]));
                }
                _mm_storeu_si16(pp, _mm_cvtsi32_si128((unsigned short)float2int8_sse(_mm_mul_ps(_p, _scale))));
                pp += 2;
            }

            p0 += max_kk * A_hstep;
            pd += 2;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            pp0 += (max_kk + (max_kk >= 4 ? 4 : 0)) * 2;
#else
            pp0 += max_kk * 2;
#endif
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + i + ii;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float* ptrA = p0 + kk * A_hstep;
                float v0 = ptrA[0];
                float v1 = ptrA[1];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k0 + kk];
                    v0 *= s;
                    v1 *= s;
                }
                absmax0 = std::max(absmax0, (float)fabsf(v0));
                absmax1 = std::max(absmax1, (float)fabsf(v1));
            }

            float scale0 = 0.f;
            float scale1 = 0.f;
            if (absmax0 != 0.f)
            {
                scale0 = 127.f / absmax0;
            }
            if (absmax1 != 0.f)
            {
                scale1 = 127.f / absmax1;
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;

            signed char* pp = pp0;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float* ptrA = p0 + kk * A_hstep;
                float v0 = ptrA[0];
                float v1 = ptrA[1];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k0 + kk];
                    v0 *= s;
                    v1 *= s;
                }
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
            }

            p0 += max_kk * A_hstep;
            pp0 += max_kk * 2;
            pd += 2;
        }
    }

#if __SSE2__
#if __AVX2__
#if __AVX512F__
    __m512i _vindex512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    _vindex512 = _mm512_mullo_epi32(_vindex512, _mm512_set1_epi32((int)A_hstep));
#endif // __AVX512F__
    __m256i _vindex256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    _vindex256 = _mm256_mullo_epi32(_vindex256, _mm256_set1_epi32((int)A_hstep));
#endif // __AVX2__
#endif // __SSE2__

    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + i + ii;
        signed char* pp0 = outptr + ii * out_hstep;
        float* pd = descale_ptr + ii * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            signed char* pp = pp0;
            float absmax = 0.f;

            int kk = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
            __m512 _absmax512 = _mm512_setzero_ps();
            for (; kk + 15 < max_kk; kk += 16)
            {
                const float* ptrA = p0 + kk * A_hstep;
                __m512 _p = _mm512_i32gather_ps(_vindex512, ptrA, sizeof(float));
                if (input_scale_ptr)
                    _p = _mm512_mul_ps(_p, _mm512_loadu_ps(input_scale_ptr + k0 + kk));
                _absmax512 = _mm512_max_ps(_absmax512, abs512_ps(_p));
            }
            absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax512));
#endif // __AVX512F__
            __m256 _absmax256 = _mm256_setzero_ps();
            for (; kk + 7 < max_kk; kk += 8)
            {
                const float* ptrA = p0 + kk * A_hstep;
                __m256 _p = _mm256_i32gather_ps(ptrA, _vindex256, sizeof(float));
                if (input_scale_ptr)
                    _p = _mm256_mul_ps(_p, _mm256_loadu_ps(input_scale_ptr + k0 + kk));
                _absmax256 = _mm256_max_ps(_absmax256, abs256_ps(_p));
            }
            absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax256));
#endif // __AVX2__
            __m128 _absmax128 = _mm_setzero_ps();
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float* ptrA = p0 + kk * A_hstep;
                __m128 _p = _mm_setr_ps(ptrA[0], ptrA[A_hstep], ptrA[A_hstep * 2], ptrA[A_hstep * 3]);
                if (input_scale_ptr)
                    _p = _mm_mul_ps(_p, _mm_loadu_ps(input_scale_ptr + k0 + kk));
                _absmax128 = _mm_max_ps(_absmax128, abs_ps(_p));
            }
            absmax = std::max(absmax, _mm_reduce_max_ps(_absmax128));
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                float v = p0[kk * A_hstep];
                if (input_scale_ptr)
                    v *= input_scale_ptr[k0 + kk];
                absmax = std::max(absmax, (float)fabsf(v));
            }

            if (absmax == 0.f)
            {
                pd[0] = 0.f;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                memset(pp, 0, max_kk >= 4 ? max_kk + 4 : max_kk);
                pp0 += max_kk + (max_kk >= 4 ? 4 : 0);
#else
                memset(pp, 0, max_kk);
                pp0 += max_kk;
#endif
                p0 += max_kk * A_hstep;
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
#if __AVX2__
#if __AVX512F__
            __m512 _scale512 = _mm512_set1_ps(scale);
            for (; kk + 15 < max_kk; kk += 16)
            {
                const float* ptrA = p0 + kk * A_hstep;
                __m512 _p = _mm512_i32gather_ps(_vindex512, ptrA, sizeof(float));
                if (input_scale_ptr)
                {
                    _p = _mm512_mul_ps(_p, _mm512_loadu_ps(input_scale_ptr + k0 + kk));
                }
                __m128i _q = float2int8_avx512(_mm512_mul_ps(_p, _scale512));
                _mm_storeu_si128((__m128i*)pp, _q);
                pp += 16;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m256i _q16 = _mm256_cvtepi8_epi16(_q);
                __m256i _q32 = _mm256_madd_epi16(_q16, _mm256_set1_epi16(1));
                w_shift += _mm_reduce_add_epi32(_mm256_castsi256_si128(_q32));
                w_shift += _mm_reduce_add_epi32(_mm256_extracti128_si256(_q32, 1));
#endif
            }
#endif // __AVX512F__
            __m256 _scale256 = _mm256_set1_ps(scale);
            for (; kk + 7 < max_kk; kk += 8)
            {
                const float* ptrA = p0 + kk * A_hstep;
                __m256 _p = _mm256_i32gather_ps(ptrA, _vindex256, sizeof(float));
                if (input_scale_ptr)
                {
                    _p = _mm256_mul_ps(_p, _mm256_loadu_ps(input_scale_ptr + k0 + kk));
                }
                const int64_t q = float2int8_avx(_mm256_mul_ps(_p, _scale256));
                _mm_storeu_si64(pp, _mm_cvtsi64_si128(q));
                pp += 8;
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
            __m128 _scale128 = _mm_set1_ps(scale);
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float* ptrA = p0 + kk * A_hstep;
                __m128 _p = _mm_setr_ps(ptrA[0], ptrA[A_hstep], ptrA[A_hstep * 2], ptrA[A_hstep * 3]);
                if (input_scale_ptr)
                {
                    _p = _mm_mul_ps(_p, _mm_loadu_ps(input_scale_ptr + k0 + kk));
                }
                const int32_t q = float2int8_sse(_mm_mul_ps(_p, _scale128));
                _mm_storeu_si32(pp, _mm_cvtsi32_si128(q));
                pp += 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                __m128i _q8 = _mm_cvtsi32_si128(q);
                __m128i _q16 = _mm_unpacklo_epi8(_q8, _mm_cmpgt_epi8(_mm_setzero_si128(), _q8));
                w_shift += _mm_reduce_add_epi32(_mm_madd_epi16(_q16, _mm_set1_epi16(1)));
#endif
            }
#endif // __SSE2__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _mm_storeu_si32(pp, _mm_cvtsi32_si128(w_shift * 127));
                pp += 4;
            }
#endif
            for (; kk < max_kk; kk++)
            {
                float v = p0[kk * A_hstep];
                if (input_scale_ptr)
                {
                    v *= input_scale_ptr[k0 + kk];
                }
                *pp++ = float2int8(v * scale);
            }

            p0 += max_kk * A_hstep;
            pd++;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            pp0 += max_kk + (max_kk >= 4 ? 4 : 0);
#else
            pp0 += max_kk;
#endif
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        gemm_transB_packed_tile_wq_int8_avx512vnni(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        gemm_transB_packed_tile_wq_int8_avxvnniint8(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        gemm_transB_packed_tile_wq_int8_avxvnni(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        gemm_transB_packed_tile_wq_int8_avx2(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_xop())
    {
        gemm_transB_packed_tile_wq_int8_xop(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

    const signed char* pAT = AT_tile;
    const int A_hstep = AT_tile.w;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = AT_descales_tile.w;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int tile_K = max_kk;
    const int block_count = (K + block_size - 1) / block_size;
    const int block_start = k / block_size;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)k * 8;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m512i _sum0 = _mm512_setzero_si512();
                __m512i _sum1 = _mm512_setzero_si512();
                __m512i _sum2 = _mm512_setzero_si512();
                __m512i _sum3 = _mm512_setzero_si512();
                __m512i _sum4 = _mm512_setzero_si512();
                __m512i _sum5 = _mm512_setzero_si512();
                __m512i _sum6 = _mm512_setzero_si512();
                __m512i _sum7 = _mm512_setzero_si512();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
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
                    __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                    __m512i _pB0 = combine8x2_epi32(_pB, _pB);
                    __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _pB3 = _mm512_alignr_epi8(_pB2, _pB2, 4);
                    _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                    _sum4 = _mm512_dpbusd_epi32(_sum4, _pB2, _pA0);
                    _sum5 = _mm512_dpbusd_epi32(_sum5, _pB3, _pA0);
                    _sum6 = _mm512_dpbusd_epi32(_sum6, _pB2, _pA1);
                    _sum7 = _mm512_dpbusd_epi32(_sum7, _pB3, _pA1);
                    pB += 32;
                    pA += 64;
                }
                if (max_kk0 >= 4)
                {
                    __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _w_shift1 = _mm512_alignr_epi8(_w_shift0, _w_shift0, 8);
                    _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                    _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                    _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                    _sum4 = _mm512_sub_epi32(_sum4, _w_shift0);
                    _sum5 = _mm512_sub_epi32(_sum5, _w_shift0);
                    _sum6 = _mm512_sub_epi32(_sum6, _w_shift1);
                    _sum7 = _mm512_sub_epi32(_sum7, _w_shift1);
                    pA += 64;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                    __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                    __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
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
                    pB += 16;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);
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
                    pB += 8;
                    pA += 16;
                }

                __m512 _ad0 = _mm512_loadu_ps(pA_descales);
                __m512 _ad1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_ad0), _mm512_castps_si512(_ad0), 8));
                __m256 _b = _mm256_loadu_ps(pB_descales);
                __m512 _bd0 = combine8x2_ps(_b, _b);
                __m512 _bd1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_bd0), _mm512_castps_si512(_bd0), 4));
                __m512 _bd2 = _mm512_castsi512_ps(_mm512_permutex_epi64(_mm512_castps_si512(_bd0), _MM_SHUFFLE(1, 0, 3, 2)));
                __m512 _bd3 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_bd2), _mm512_castps_si512(_bd2), 4));
                _fsum0 = _mm512_add_ps(_fsum0, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum0), _mm512_mul_ps(_ad0, _bd0)));
                _fsum1 = _mm512_add_ps(_fsum1, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum1), _mm512_mul_ps(_ad0, _bd1)));
                _fsum2 = _mm512_add_ps(_fsum2, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum2), _mm512_mul_ps(_ad1, _bd0)));
                _fsum3 = _mm512_add_ps(_fsum3, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum3), _mm512_mul_ps(_ad1, _bd1)));
                _fsum4 = _mm512_add_ps(_fsum4, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum4), _mm512_mul_ps(_ad0, _bd2)));
                _fsum5 = _mm512_add_ps(_fsum5, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum5), _mm512_mul_ps(_ad0, _bd3)));
                _fsum6 = _mm512_add_ps(_fsum6, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum6), _mm512_mul_ps(_ad1, _bd2)));
                _fsum7 = _mm512_add_ps(_fsum7, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum7), _mm512_mul_ps(_ad1, _bd3)));
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
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)k * 4;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m512i _sum0 = _mm512_setzero_si512();
                __m512i _sum1 = _mm512_setzero_si512();
                __m512i _sum2 = _mm512_setzero_si512();
                __m512i _sum3 = _mm512_setzero_si512();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pB));
                    __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                    pB += 16;
                    pA += 64;
                }
                if (max_kk0 >= 4)
                {
                    __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _w_shift1 = _mm512_alignr_epi8(_w_shift0, _w_shift0, 8);
                    _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                    _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                    _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                    pA += 64;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                    __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));
                    __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm512_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pB += 8;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_mm_castps_si128(_mm_load1_ps((const float*)pB)));
                    __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0)));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1)));
                    pB += 4;
                    pA += 16;
                }

                __m512 _ad0 = _mm512_loadu_ps(pA_descales);
                __m512 _ad1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_ad0), _mm512_castps_si512(_ad0), 8));
                __m512 _bd0 = _mm512_broadcast_f32x4(_mm_loadu_ps(pB_descales));
                __m512 _bd1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_bd0), _mm512_castps_si512(_bd0), 4));
                _fsum0 = _mm512_add_ps(_fsum0, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum0), _mm512_mul_ps(_ad0, _bd0)));
                _fsum1 = _mm512_add_ps(_fsum1, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum1), _mm512_mul_ps(_ad0, _bd1)));
                _fsum2 = _mm512_add_ps(_fsum2, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum2), _mm512_mul_ps(_ad1, _bd0)));
                _fsum3 = _mm512_add_ps(_fsum3, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum3), _mm512_mul_ps(_ad1, _bd1)));
                pA_descales += 16;
                pB_descales += 4;
            }

            _mm512_storeu_ps(outptr + 0, _fsum0);
            _mm512_storeu_ps(outptr + 16, _fsum1);
            _mm512_storeu_ps(outptr + 32, _fsum2);
            _mm512_storeu_ps(outptr + 48, _fsum3);
            outptr += 64;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)k * 2;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m512i _sum0 = _mm512_setzero_si512();
                __m512i _sum1 = _mm512_setzero_si512();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB0 = _mm512_set1_epi64(_mm_cvtsi128_si64(_mm_loadu_si64(pB)));
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                    pB += 8;
                    pA += 64;
                }
                if (max_kk0 >= 4)
                {
                    __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                    _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                    pA += 64;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                    __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));
                    __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);
                    __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    pB += 4;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m128i _pB = _mm_set1_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(pB)));
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);
                    __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                    pB += 2;
                    pA += 16;
                }

                __m512 _ad0 = _mm512_loadu_ps(pA_descales);
                __m512 _bd0 = _mm512_castsi512_ps(_mm512_set1_epi64(_mm_cvtsi128_si64(_mm_loadu_si64(pB_descales))));
                __m512 _bd1 = _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(_bd0), _mm512_castps_si512(_bd0), 4));
                _fsum0 = _mm512_add_ps(_fsum0, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum0), _mm512_mul_ps(_ad0, _bd0)));
                _fsum1 = _mm512_add_ps(_fsum1, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum1), _mm512_mul_ps(_ad0, _bd1)));
                pA_descales += 16;
                pB_descales += 2;
            }

            _mm512_storeu_ps(outptr + 0, _fsum0);
            _mm512_storeu_ps(outptr + 16, _fsum1);
            outptr += 32;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + (size_t)k;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m512i _sum0 = _mm512_setzero_si512();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm_loadu_si32(pB)));
                    _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                    pB += 4;
                    pA += 64;
                }
                if (max_kk0 >= 4)
                {
                    __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                    _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                    pA += 64;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                    __m512i _pB0 = _mm512_cvtepi8_epi16(_mm256_set1_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(pB))));
                    _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    pB += 2;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m512i _pA0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)pA));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_mullo_epi32(_pA0, _mm512_set1_epi32(pB[0])));
                    pB += 1;
                    pA += 16;
                }

                __m512 _ad0 = _mm512_loadu_ps(pA_descales);
                __m512 _bd0 = _mm512_set1_ps(pB_descales[0]);
                _fsum0 = _mm512_add_ps(_fsum0, _mm512_mul_ps(_mm512_cvtepi32_ps(_sum0), _mm512_mul_ps(_ad0, _bd0)));
                pA_descales += 16;
                pB_descales += 1;
            }

            _mm512_storeu_ps(outptr + 0, _fsum0);
            outptr += 16;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 16;
        pAT_descales += A_descales_hstep * 16;
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)k * 8;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                __m256i _sum2 = _mm256_setzero_si256();
                __m256i _sum3 = _mm256_setzero_si256();
                __m256i _sum4 = _mm256_setzero_si256();
                __m256i _sum5 = _mm256_setzero_si256();
                __m256i _sum6 = _mm256_setzero_si256();
                __m256i _sum7 = _mm256_setzero_si256();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pB);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB3 = _mm256_alignr_epi8(_pB2, _pB2, 4);
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _pB1, _pA1);
                    _sum4 = _mm256_comp_dpbusd_epi32(_sum4, _pB2, _pA0);
                    _sum5 = _mm256_comp_dpbusd_epi32(_sum5, _pB3, _pA0);
                    _sum6 = _mm256_comp_dpbusd_epi32(_sum6, _pB2, _pA1);
                    _sum7 = _mm256_comp_dpbusd_epi32(_sum7, _pB3, _pA1);
                    pB += 32;
                    pA += 32;
                }
                if (max_kk0 >= 4)
                {
                    __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _w_shift1 = _mm256_alignr_epi8(_w_shift0, _w_shift0, 8);
                    _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _w_shift0);
                    _sum2 = _mm256_sub_epi32(_sum2, _w_shift1);
                    _sum3 = _mm256_sub_epi32(_sum3, _w_shift1);
                    _sum4 = _mm256_sub_epi32(_sum4, _w_shift0);
                    _sum5 = _mm256_sub_epi32(_sum5, _w_shift0);
                    _sum6 = _mm256_sub_epi32(_sum6, _w_shift1);
                    _sum7 = _mm256_sub_epi32(_sum7, _w_shift1);
                    pA += 32;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB8 = _mm_loadu_si128((const __m128i*)pB);
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
                    pB += 16;
                    pA += 16;
                }
                if (kk < max_kk0)
                {
                    __m128i _pA0 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB0 = _mm_loadl_epi64((const __m128i*)pB);
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
                    pB += 8;
                }

                __m256 _ad0 = _mm256_loadu_ps(pA_descales);
                __m256 _ad1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_ad0), _mm256_castps_si256(_ad0), 8));
                __m256 _bd0 = _mm256_loadu_ps(pB_descales);
                __m256 _bd1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_bd0), _mm256_castps_si256(_bd0), 4));
                __m256 _bd2 = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(_bd0), _MM_SHUFFLE(1, 0, 3, 2)));
                __m256 _bd3 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_bd2), _mm256_castps_si256(_bd2), 4));
                _fsum0 = _mm256_add_ps(_fsum0, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_ad0, _bd0)));
                _fsum1 = _mm256_add_ps(_fsum1, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_ad0, _bd1)));
                _fsum2 = _mm256_add_ps(_fsum2, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum2), _mm256_mul_ps(_ad1, _bd0)));
                _fsum3 = _mm256_add_ps(_fsum3, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum3), _mm256_mul_ps(_ad1, _bd1)));
                _fsum4 = _mm256_add_ps(_fsum4, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum4), _mm256_mul_ps(_ad0, _bd2)));
                _fsum5 = _mm256_add_ps(_fsum5, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum5), _mm256_mul_ps(_ad0, _bd3)));
                _fsum6 = _mm256_add_ps(_fsum6, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum6), _mm256_mul_ps(_ad1, _bd2)));
                _fsum7 = _mm256_add_ps(_fsum7, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum7), _mm256_mul_ps(_ad1, _bd3)));
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
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)k * 4;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                __m256i _sum2 = _mm256_setzero_si256();
                __m256i _sum3 = _mm256_setzero_si256();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                    __m256i _pB0 = combine4x2_epi32(_pB, _pB);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
#if __AVXVNNIINT8__
                    _sum0 = _mm256_dpbssd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm256_dpbssd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm256_dpbssd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm256_dpbssd_epi32(_sum3, _pB1, _pA1);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _pB1, _pA1);
#endif // __AVXVNNIINT8__
                    pB += 16;
                    pA += 32;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _w_shift1 = _mm256_alignr_epi8(_w_shift0, _w_shift0, 8);
                    _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _w_shift0);
                    _sum2 = _mm256_sub_epi32(_sum2, _w_shift1);
                    _sum3 = _mm256_sub_epi32(_sum3, _w_shift1);
                    pA += 32;
                }
#endif
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = _mm_castpd_si128(_mm_load1_pd((const double*)pB));
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pB += 8;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA0 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB0 = _mm_castps_si128(_mm_load1_ps((const float*)pB));
                    _pA0 = _mm_cvtepi8_epi16(_pA0);
                    _pB0 = _mm_cvtepi8_epi16(_pB0);
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0)));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1)));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0)));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1)));
                    pB += 4;
                    pA += 8;
                }

                __m256 _ad0 = _mm256_loadu_ps(pA_descales);
                __m256 _ad1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_ad0), _mm256_castps_si256(_ad0), 8));
                __m128 _b = _mm_loadu_ps(pB_descales);
                __m256 _bd0 = combine4x2_ps(_b, _b);
                __m256 _bd1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_bd0), _mm256_castps_si256(_bd0), 4));
                _fsum0 = _mm256_add_ps(_fsum0, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_ad0, _bd0)));
                _fsum1 = _mm256_add_ps(_fsum1, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_ad0, _bd1)));
                _fsum2 = _mm256_add_ps(_fsum2, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum2), _mm256_mul_ps(_ad1, _bd0)));
                _fsum3 = _mm256_add_ps(_fsum3, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum3), _mm256_mul_ps(_ad1, _bd1)));
                pA_descales += 8;
                pB_descales += 4;
            }

            _mm256_storeu_ps(outptr + 0, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            _mm256_storeu_ps(outptr + 16, _fsum2);
            _mm256_storeu_ps(outptr + 24, _fsum3);
            outptr += 32;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)k * 2;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pB0 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
#if __AVXVNNIINT8__
                    _sum0 = _mm256_dpbssd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm256_dpbssd_epi32(_sum1, _pB1, _pA0);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB1, _pA0);
#endif // __AVXVNNIINT8__
                    pB += 8;
                    pA += 32;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                    _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _w_shift0);
                    pA += 32;
                }
#endif
#else
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);
                    __m256i _pA01 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(_pA));
                    __m256i _pA23 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_pA, 1));
                    __m256i _pB01 = _mm256_cvtepi8_epi16(_mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 0, 0, 0)));
                    __m256i _pB23 = _mm256_cvtepi8_epi16(_mm_shuffle_epi32(_pB, _MM_SHUFFLE(1, 1, 1, 1)));
                    __m256i _pB01_1 = _mm256_alignr_epi8(_pB01, _pB01, 4);
                    __m256i _pB23_1 = _mm256_alignr_epi8(_pB23, _pB23, 4);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA01, _pB01);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA01, _pB01_1);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA23, _pB23);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA23, _pB23_1);
                    pB += 8;
                    pA += 32;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    pB += 4;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB0 = _mm_set1_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(pB)));
                    _pA = _mm_cvtepi8_epi16(_pA);
                    _pB0 = _mm_cvtepi8_epi16(_pB0);
                    __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0)));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1)));
                    pB += 2;
                    pA += 8;
                }

                __m256 _ad0 = _mm256_loadu_ps(pA_descales);
                __m256 _bd0 = _mm256_castpd_ps(_mm256_broadcast_sd((const double*)pB_descales));
                __m256 _bd1 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(_bd0), _mm256_castps_si256(_bd0), 4));
                _fsum0 = _mm256_add_ps(_fsum0, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_ad0, _bd0)));
                _fsum1 = _mm256_add_ps(_fsum1, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_ad0, _bd1)));
                pA_descales += 8;
                pB_descales += 2;
            }

            _mm256_storeu_ps(outptr + 0, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            outptr += 16;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + (size_t)k;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pB0 = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));
#if __AVXVNNIINT8__
                    _sum0 = _mm256_dpbssd_epi32(_sum0, _pB0, _pA0);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB0, _pA0);
#endif // __AVXVNNIINT8__
                    pB += 4;
                    pA += 32;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                    _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                    pA += 32;
                }
#endif
#else
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pA01 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(_pA));
                    __m256i _pA23 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_pA, 1));
                    __m128i _pB16 = _mm_cvtepi8_epi16(_mm_loadu_si32(pB));
                    __m256i _pB01 = _mm256_broadcastsi128_si256(_mm_shuffle_epi32(_pB16, _MM_SHUFFLE(0, 0, 0, 0)));
                    __m256i _pB23 = _mm256_broadcastsi128_si256(_mm_shuffle_epi32(_pB16, _MM_SHUFFLE(1, 1, 1, 1)));
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA01, _pB01);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA23, _pB23);
                    pB += 4;
                    pA += 32;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_mm_set1_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(pB))));
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    pB += 2;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                    _pA = _mm_cvtepi8_epi16(_pA);
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _mm_set1_epi16(pB[0]))));
                    pB += 1;
                    pA += 8;
                }

                __m256 _ad0 = _mm256_loadu_ps(pA_descales);
                _fsum0 = _mm256_add_ps(_fsum0, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_ad0, _mm256_set1_ps(pB_descales[0]))));
                pA_descales += 8;
                pB_descales++;
            }

            _mm256_storeu_ps(outptr, _fsum0);
            outptr += 8;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 8;
        pAT_descales += A_descales_hstep * 8;
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)k * 8;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                __m256i _sum2 = _mm256_setzero_si256();
                __m256i _sum3 = _mm256_setzero_si256();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)pA));
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pB);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _pB1, _pA1);
                    pA += 16;
                    pB += 32;
                }
                if (max_kk0 >= 4)
                {
                    __m256i _w_shift0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)pA));
                    __m256i _w_shift1 = _mm256_alignr_epi8(_w_shift0, _w_shift0, 8);
                    _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _w_shift0);
                    _sum2 = _mm256_sub_epi32(_sum2, _w_shift1);
                    _sum3 = _mm256_sub_epi32(_sum3, _w_shift1);
                    pA += 16;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8x1 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pA8 = _mm_unpacklo_epi64(_pA8x1, _pA8x1);
                    __m128i _pB8 = _mm_loadu_si128((const __m128i*)pB);
                    __m256i _pA0 = _mm256_cvtepi8_epi16(_pA8);
                    __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                    __m256i _pB0 = _mm256_cvtepi8_epi16(_pB8);
                    __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pA += 8;
                    pB += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pB8 = _mm_loadl_epi64((const __m128i*)pB);
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
                    pB += 8;
                }

                __m128 _ad128 = _mm_loadu_ps(pA_descales);
                __m256 _ad0 = combine4x2_ps(_ad128, _ad128);
                __m256 _ad1 = _mm256_shuffle_ps(_ad0, _ad0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _bd0 = _mm256_loadu_ps(pB_descales);
                __m256 _bd1 = _mm256_shuffle_ps(_bd0, _bd0, _MM_SHUFFLE(0, 3, 2, 1));
                _fsum0 = _mm256_add_ps(_fsum0, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_ad0, _bd0)));
                _fsum1 = _mm256_add_ps(_fsum1, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_ad0, _bd1)));
                _fsum2 = _mm256_add_ps(_fsum2, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum2), _mm256_mul_ps(_ad1, _bd0)));
                _fsum3 = _mm256_add_ps(_fsum3, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum3), _mm256_mul_ps(_ad1, _bd1)));
                pA_descales += 4;
                pB_descales += 8;
            }

            _mm256_storeu_ps(outptr, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            _mm256_storeu_ps(outptr + 16, _fsum2);
            _mm256_storeu_ps(outptr + 24, _fsum3);
            outptr += 32;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)k * 4;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pA1 = _mm_alignr_epi8(_pA0, _pA0, 8);
                    __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                    __m128i _pB1 = _mm_alignr_epi8(_pB0, _pB0, 4);
#if __AVXVNNIINT8__
                    _sum0 = _mm_dpbssd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm_dpbssd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm_dpbssd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm_dpbssd_epi32(_sum3, _pB1, _pA1);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm_comp_dpbusd_epi32(_sum1, _pB1, _pA0);
                    _sum2 = _mm_comp_dpbusd_epi32(_sum2, _pB0, _pA1);
                    _sum3 = _mm_comp_dpbusd_epi32(_sum3, _pB1, _pA1);
#endif // __AVXVNNIINT8__
                    pA += 16;
                    pB += 16;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    __m128i _w_shift0 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _w_shift1 = _mm_alignr_epi8(_w_shift0, _w_shift0, 8);
                    _sum0 = _mm_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm_sub_epi32(_sum1, _w_shift0);
                    _sum2 = _mm_sub_epi32(_sum2, _w_shift1);
                    _sum3 = _mm_sub_epi32(_sum3, _w_shift1);
                    pA += 16;
                }
#endif
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB8 = _mm_loadl_epi64((const __m128i*)pB);
                    __m128i _pA0 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB0 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pB8 = _mm_loadu_si32(pB);
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
                    pB += 4;
                }

                __m128 _ad0 = _mm_loadu_ps(pA_descales);
                __m128 _ad1 = _mm_shuffle_ps(_ad0, _ad0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128 _bd0 = _mm_loadu_ps(pB_descales);
                __m128 _bd1 = _mm_shuffle_ps(_bd0, _bd0, _MM_SHUFFLE(0, 3, 2, 1));
                _fsum0 = _mm_add_ps(_fsum0, _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _mm_mul_ps(_ad0, _bd0)));
                _fsum1 = _mm_add_ps(_fsum1, _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _mm_mul_ps(_ad0, _bd1)));
                _fsum2 = _mm_add_ps(_fsum2, _mm_mul_ps(_mm_cvtepi32_ps(_sum2), _mm_mul_ps(_ad1, _bd0)));
                _fsum3 = _mm_add_ps(_fsum3, _mm_mul_ps(_mm_cvtepi32_ps(_sum3), _mm_mul_ps(_ad1, _bd1)));
                pA_descales += 4;
                pB_descales += 4;
            }

            _mm_storeu_ps(outptr, _fsum0);
            _mm_storeu_ps(outptr + 4, _fsum1);
            _mm_storeu_ps(outptr + 8, _fsum2);
            _mm_storeu_ps(outptr + 12, _fsum3);
            outptr += 16;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)k * 2;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB8 = _mm_loadl_epi64((const __m128i*)pB);
                    __m128i _pB0 = _mm_unpacklo_epi64(_pB8, _pB8);
                    __m128i _pB1 = _mm_alignr_epi8(_pB0, _pB0, 4);
#if __AVXVNNIINT8__
                    _sum0 = _mm_dpbssd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm_dpbssd_epi32(_sum1, _pB1, _pA0);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pB0, _pA0);
                    _sum1 = _mm_comp_dpbusd_epi32(_sum1, _pB1, _pA0);
#endif // __AVXVNNIINT8__
                    pA += 16;
                    pB += 8;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    __m128i _w_shift0 = _mm_loadu_si128((const __m128i*)pA);
                    _sum0 = _mm_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm_sub_epi32(_sum1, _w_shift0);
                    pA += 16;
                }
#endif
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB8 = _mm_castps_si128(_mm_load1_ps((const float*)pB));
                    __m128i _pA0 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB0 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pB8 = _mm_set1_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(pB)));
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pA0 = _mm_unpacklo_epi16(_pA16, _pA16);
                    __m128i _pB0 = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                    pA += 4;
                    pB += 2;
                }

                __m128 _ad = _mm_loadu_ps(pA_descales);
                __m128 _bd0 = _mm_setr_ps(pB_descales[0], pB_descales[1], pB_descales[0], pB_descales[1]);
                __m128 _bd1 = _mm_shuffle_ps(_bd0, _bd0, _MM_SHUFFLE(0, 3, 2, 1));
                _fsum0 = _mm_add_ps(_fsum0, _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _mm_mul_ps(_ad, _bd0)));
                _fsum1 = _mm_add_ps(_fsum1, _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _mm_mul_ps(_ad, _bd1)));
                pA_descales += 4;
                pB_descales += 2;
            }

            _mm_storeu_ps(outptr, _fsum0);
            _mm_storeu_ps(outptr + 4, _fsum1);
            outptr += 8;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + (size_t)k;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = _mm_set1_epi32(_mm_cvtsi128_si32(_mm_loadu_si32(pB)));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pB, _pA);
#endif // __AVXVNNIINT8__
                    pA += 16;
                    pB += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _sum = _mm_sub_epi32(_sum, _mm_loadu_si128((const __m128i*)pA));
                    pA += 16;
                }
#endif
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB8 = _mm_set1_epi16((short)_mm_cvtsi128_si32(_mm_loadu_si16(pB)));
                    __m128i _pA = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 8;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_unpacklo_epi16(_pA16, _mm_setzero_si128());
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _mm_set1_epi16(pB[0]));
                    pA += 4;
                    pB++;
                }

                __m128 _ad = _mm_loadu_ps(pA_descales);
                _fsum = _mm_add_ps(_fsum, _mm_mul_ps(_mm_cvtepi32_ps(_sum), _mm_mul_ps(_ad, _mm_set1_ps(pB_descales[0]))));
                pA_descales += 4;
                pB_descales++;
            }

            _mm_storeu_ps(outptr, _fsum);
            outptr += 4;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 4;
        pAT_descales += A_descales_hstep * 4;
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)k * 8;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pA128 = _mm_unpacklo_epi64(_pA8, _pA8);
                    __m256i _pA0 = _mm256_broadcastsi128_si256(_pA128);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                    _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB, _pA0);
                    _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB, _pA1);
                    pA += 8;
                    pB += 32;
                }
                if (max_kk0 >= 4)
                {
                    __m128i _w_shift64 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _w_shift128 = _mm_unpacklo_epi64(_w_shift64, _w_shift64);
                    __m256i _w_shift0 = _mm256_broadcastsi128_si256(_w_shift128);
                    __m256i _w_shift1 = _mm256_shuffle_epi32(_w_shift0, _MM_SHUFFLE(2, 3, 0, 1));
                    _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                    _sum1 = _mm256_sub_epi32(_sum1, _w_shift1);
                    pA += 8;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pA16x1 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA16 = _mm_unpacklo_epi64(_pA16x1, _pA16x1);
                    __m256i _pA0 = _mm256_broadcastsi128_si256(_pA16);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)pB));
                    _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB);
                    _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA1, _pB);
                    pA += 4;
                    pB += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_loadu_si16(pA);
                    __m128i _pA32x1 = _mm_cvtepi8_epi32(_pA8);
                    __m128i _pA128 = _mm_shuffle_epi32(_pA32x1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256i _pA0 = _mm256_broadcastsi128_si256(_pA128);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)pB));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_mullo_epi32(_pA0, _pB));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_mullo_epi32(_pA1, _pB));
                    pA += 2;
                    pB += 8;
                }

                __m128 _ad2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pA_descales);
                __m128 _ad128 = _mm_movelh_ps(_ad2, _ad2);
                __m256 _ad0 = combine4x2_ps(_ad128, _ad128);
                __m256 _ad1 = _mm256_shuffle_ps(_ad0, _ad0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256 _bd = _mm256_loadu_ps(pB_descales);
                _fsum0 = _mm256_add_ps(_fsum0, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum0), _mm256_mul_ps(_ad0, _bd)));
                _fsum1 = _mm256_add_ps(_fsum1, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum1), _mm256_mul_ps(_ad1, _bd)));
                pA_descales += 2;
                pB_descales += 8;
            }

            _mm256_storeu_ps(outptr, _fsum0);
            _mm256_storeu_ps(outptr + 8, _fsum1);
            outptr += 16;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)k * 4;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pA = _mm_unpacklo_epi64(_pA8, _pA8);
                    __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
#if __AVXVNNIINT8__
                    _sum0 = _mm_dpbssd_epi32(_sum0, _pB0, _pA);
                    _sum1 = _mm_dpbssd_epi32(_sum1, _pB1, _pA);
#else  // __AVXVNNIINT8__
                    _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pB0, _pA);
                    _sum1 = _mm_comp_dpbusd_epi32(_sum1, _pB1, _pA);
#endif // __AVXVNNIINT8__
                    pA += 8;
                    pB += 16;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    __m128i _w_shift64 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _w_shift = _mm_unpacklo_epi64(_w_shift64, _w_shift64);
                    _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                    _sum1 = _mm_sub_epi32(_sum1, _w_shift);
                    pA += 8;
                }
#endif
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pA16x1 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_unpacklo_epi64(_pA16x1, _pA16x1);
                    __m128i _pB8 = _mm_loadl_epi64((const __m128i*)pB);
                    __m128i _pB0 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA, _pB1);
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_loadu_si16(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA32x1 = _mm_unpacklo_epi16(_pA16, _pA16);
                    __m128i _pA = _mm_shuffle_epi32(_pA32x1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m128i _pB8 = _mm_loadu_si32(pB);
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB0 = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA, _pB0);
                    _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA, _pB1);
                    pA += 2;
                    pB += 4;
                }

                __m128 _ad2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pA_descales);
                __m128 _ad = _mm_movelh_ps(_ad2, _ad2);
                __m128 _bd0 = _mm_loadu_ps(pB_descales);
                __m128 _bd1 = _mm_shuffle_ps(_bd0, _bd0, _MM_SHUFFLE(0, 3, 2, 1));
                _fsum0 = _mm_add_ps(_fsum0, _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _mm_mul_ps(_ad, _bd0)));
                _fsum1 = _mm_add_ps(_fsum1, _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _mm_mul_ps(_ad, _bd1)));
                pA_descales += 2;
                pB_descales += 4;
            }

            _mm_storeu_ps(outptr, _fsum0);
            _mm_storeu_ps(outptr + 4, _fsum1);
            outptr += 8;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)k * 2;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 2;
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

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pA = _mm_unpacklo_epi32(_pA8, _pA8);
                    __m128i _pB8 = _mm_loadl_epi64((const __m128i*)pB);
                    __m128i _pB = _mm_unpacklo_epi64(_pB8, _pB8);
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pB, _pA);
#endif // __AVXVNNIINT8__
                    pA += 8;
                    pB += 8;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    __m128i _w_shift64 = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _w_shift = _mm_unpacklo_epi32(_w_shift64, _w_shift64);
                    _sum = _mm_sub_epi32(_sum, _w_shift);
                    pA += 8;
                }
#endif
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pA16x1 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_unpacklo_epi32(_pA16x1, _pA16x1);
                    __m128i _pB8 = _mm_loadu_si32(pB);
                    __m128i _pB16x1 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB = _mm_unpacklo_epi64(_pB16x1, _pB16x1);
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_loadu_si16(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA32x1 = _mm_unpacklo_epi16(_pA16, _pA16);
                    __m128i _pA = _mm_unpacklo_epi32(_pA32x1, _pA32x1);
                    __m128i _pB8 = _mm_loadu_si16(pB);
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB32x1 = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    __m128i _pB = _mm_unpacklo_epi64(_pB32x1, _pB32x1);
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 2;
                }

                __m128 _ad2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pA_descales);
                __m128 _ad = _mm_unpacklo_ps(_ad2, _ad2);
                __m128 _bd2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pB_descales);
                __m128 _bd = _mm_movelh_ps(_bd2, _bd2);
                _fsum = _mm_add_ps(_fsum, _mm_mul_ps(_mm_cvtepi32_ps(_sum), _mm_mul_ps(_ad, _bd)));
                pA_descales += 2;
                pB_descales += 2;
            }

            _mm_storeu_ps(outptr, _fsum);
            outptr += 4;

#else
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
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int b0 = pB[0];
                    int b1 = pB[1];
                    sum00 += pA[0] * b0;
                    sum01 += pA[0] * b1;
                    sum10 += pA[1] * b0;
                    sum11 += pA[1] * b1;
                    b0 = pB[2];
                    b1 = pB[3];
                    sum00 += pA[2] * b0;
                    sum01 += pA[2] * b1;
                    sum10 += pA[3] * b0;
                    sum11 += pA[3] * b1;
                    b0 = pB[4];
                    b1 = pB[5];
                    sum00 += pA[4] * b0;
                    sum01 += pA[4] * b1;
                    sum10 += pA[5] * b0;
                    sum11 += pA[5] * b1;
                    b0 = pB[6];
                    b1 = pB[7];
                    sum00 += pA[6] * b0;
                    sum01 += pA[6] * b1;
                    sum10 += pA[7] * b0;
                    sum11 += pA[7] * b1;
                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    const int b0 = pB[0];
                    const int b1 = pB[1];
                    sum00 += pA[0] * b0;
                    sum01 += pA[0] * b1;
                    sum10 += pA[1] * b0;
                    sum11 += pA[1] * b1;
                    pA += 2;
                    pB += 2;
                }

                const float ad0 = pA_descales[0];
                const float ad1 = pA_descales[1];
                fsum00 += sum00 * ad0 * pB_descales[0];
                fsum01 += sum01 * ad0 * pB_descales[1];
                fsum10 += sum10 * ad1 * pB_descales[0];
                fsum11 += sum11 * ad1 * pB_descales[1];
                pA_descales += 2;
                pB_descales += 2;
            }

            outptr[0] = fsum00;
            outptr[1] = fsum01;
            outptr[2] = fsum10;
            outptr[3] = fsum11;
            outptr += 4;
#endif // __SSE2__
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + (size_t)k;
            const float* pB_descales = pB_descales_panel + (size_t)block_start;
#if __SSE2__
            __m128 _fsum;

            if (k == 0)
            {
                _fsum = _mm_setzero_ps();
            }
            else
            {
                _fsum = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)outptr);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB8 = _mm_loadu_si32(pB);
                    __m128i _pB = _mm_shuffle_epi32(_pB8, _MM_SHUFFLE(0, 0, 0, 0));
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pB, _pA);
#endif // __AVXVNNIINT8__
                    pA += 8;
                    pB += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _sum = _mm_sub_epi32(_sum, _mm_loadl_epi64((const __m128i*)pA));
                    pA += 8;
                }
#endif
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pA = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB8 = _mm_loadu_si16(pB);
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB = _mm_shuffle_epi32(_pB16, _MM_SHUFFLE(0, 0, 0, 0));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_loadu_si16(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_unpacklo_epi16(_pA16, _pA16);
                    __m128i _pB8 = _mm_cvtsi32_si128(pB[0]);
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB32 = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    __m128i _pB = _mm_shuffle_epi32(_pB32, _MM_SHUFFLE(0, 0, 0, 0));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB++;
                }

                __m128 _ad = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pA_descales);
                _fsum = _mm_add_ps(_fsum, _mm_mul_ps(_mm_cvtepi32_ps(_sum), _mm_mul_ps(_ad, _mm_set1_ps(pB_descales[0]))));
                pA_descales += 2;
                pB_descales++;
            }

            _mm_storel_pi((__m64*)outptr, _fsum);
            outptr += 2;

#else
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
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                int sum0 = 0;
                int sum1 = 0;
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int b0 = pB[0];
                    sum0 += pA[0] * b0;
                    sum1 += pA[1] * b0;
                    b0 = pB[1];
                    sum0 += pA[2] * b0;
                    sum1 += pA[3] * b0;
                    b0 = pB[2];
                    sum0 += pA[4] * b0;
                    sum1 += pA[5] * b0;
                    b0 = pB[3];
                    sum0 += pA[6] * b0;
                    sum1 += pA[7] * b0;
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const int b0 = pB[0];
                    sum0 += pA[0] * b0;
                    sum1 += pA[1] * b0;
                    pA += 2;
                    pB++;
                }

                fsum0 += sum0 * pA_descales[0] * pB_descales[0];
                fsum1 += sum1 * pA_descales[1] * pB_descales[0];
                pA_descales += 2;
                pB_descales++;
            }

            outptr[0] = fsum0;
            outptr[1] = fsum1;
            outptr += 2;
#endif // __SSE2__
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)k * 8;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m256i _sum = _mm256_setzero_si256();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA32 = _mm_loadu_si32(pA);
                    __m256i _pA = _mm256_broadcastd_epi32(_pA32);
                    __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                    _sum = _mm256_comp_dpbusd_epi32(_sum, _pB, _pA);
                    pA += 4;
                    pB += 32;
                }
                if (max_kk0 >= 4)
                {
                    _sum = _mm256_sub_epi32(_sum, _mm256_set1_epi32(_mm_cvtsi128_si32(_mm_loadu_si32(pA))));
                    pA += 4;
                }
#else
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m256i _pA01 = _mm256_broadcastsi128_si256(_mm_shuffle_epi32(_pA16, _MM_SHUFFLE(0, 0, 0, 0)));
                    __m256i _pA23 = _mm256_broadcastsi128_si256(_mm_shuffle_epi32(_pA16, _MM_SHUFFLE(1, 1, 1, 1)));
                    __m256i _pB01 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)pB));
                    __m256i _pB23 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(pB + 16)));
                    _sum = _mm256_comp_dpwssd_epi32(_sum, _pA01, _pB01);
                    _sum = _mm256_comp_dpwssd_epi32(_sum, _pA23, _pB23);
                    pA += 4;
                    pB += 32;
                }
#endif // __AVX512VNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si16(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m256i _pA = _mm256_broadcastsi128_si256(_mm_shuffle_epi32(_pA16, _MM_SHUFFLE(0, 0, 0, 0)));
                    __m256i _pB = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)pB));
                    _sum = _mm256_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128(pA[0]);
                    __m256i _pA = _mm256_broadcastd_epi32(_pA8);
                    __m256i _pB = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)pB));
                    _sum = _mm256_add_epi32(_sum, _mm256_mullo_epi32(_pA, _pB));
                    pA++;
                    pB += 8;
                }
                __m128 _ad1 = _mm_load_ss(pA_descales);
                __m256 _ad = _mm256_broadcastss_ps(_ad1);
                __m256 _descale = _mm256_mul_ps(_ad, _mm256_loadu_ps(pB_descales));
                _fsum = _mm256_add_ps(_fsum, _mm256_mul_ps(_mm256_cvtepi32_ps(_sum), _descale));
                pA_descales += 1;
                pB_descales += 8;
            }

            _mm256_storeu_ps(outptr, _fsum);
            outptr += 8;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)k * 4;
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
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA32 = _mm_loadu_si32(pA);
                    __m128i _pA = _mm_shuffle_epi32(_pA32, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pB, _pA);
#endif // __AVXVNNIINT8__
                    pA += 4;
                    pB += 16;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _sum = _mm_sub_epi32(_sum, _mm_set1_epi32(_mm_cvtsi128_si32(_mm_loadu_si32(pA))));
                    pA += 4;
                }
#endif
#else
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA01 = _mm_shuffle_epi32(_pA16, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pA23 = _mm_shuffle_epi32(_pA16, _MM_SHUFFLE(1, 1, 1, 1));
                    __m128i _pB01x1 = _mm_loadl_epi64((const __m128i*)pB);
                    __m128i _pB23x1 = _mm_loadl_epi64((const __m128i*)(pB + 8));
                    __m128i _pB01 = _mm_unpacklo_epi8(_pB01x1, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB01x1));
                    __m128i _pB23 = _mm_unpacklo_epi8(_pB23x1, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB23x1));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA01, _pB01);
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA23, _pB23);
                    pA += 4;
                    pB += 16;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si16(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_shuffle_epi32(_pA16, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB8 = _mm_loadl_epi64((const __m128i*)pB);
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128(pA[0]);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_shuffle_epi32(_mm_unpacklo_epi16(_pA16, _pA16), _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB8 = _mm_loadu_si32(pB);
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA++;
                    pB += 4;
                }
                __m128 _ad1 = _mm_load_ss(pA_descales);
                __m128 _ad = _mm_shuffle_ps(_ad1, _ad1, _MM_SHUFFLE(0, 0, 0, 0));
                __m128 _descale = _mm_mul_ps(_ad, _mm_loadu_ps(pB_descales));
                _fsum = _mm_add_ps(_fsum, _mm_mul_ps(_mm_cvtepi32_ps(_sum), _descale));
                pA_descales += 1;
                pB_descales += 4;
            }

            _mm_storeu_ps(outptr, _fsum);
            outptr += 4;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)k * 2;
            const float* pB_descales = pB_descales_panel + (size_t)block_start * 2;
#if __SSE2__
            __m128 _fsum;

            if (k == 0)
            {
                _fsum = _mm_setzero_ps();
            }
            else
            {
                _fsum = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)outptr);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA32 = _mm_loadu_si32(pA);
                    __m128i _pA = _mm_shuffle_epi32(_pA32, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pB, _pA);
#endif // __AVXVNNIINT8__
                    pA += 4;
                    pB += 8;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _sum = _mm_sub_epi32(_sum, _mm_set1_epi32(_mm_cvtsi128_si32(_mm_loadu_si32(pA))));
                    pA += 4;
                }
#endif
#else
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA01 = _mm_shuffle_epi32(_pA16, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pA23 = _mm_shuffle_epi32(_pA16, _MM_SHUFFLE(1, 1, 1, 1));
                    __m128i _pB8 = _mm_loadl_epi64((const __m128i*)pB);
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB23 = _mm_shuffle_epi32(_pB16, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA01, _pB16);
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA23, _pB23);
                    pA += 4;
                    pB += 8;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si16(pA);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pA = _mm_shuffle_epi32(_pA16, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128i _pB8 = _mm_loadu_si32(pB);
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
                    __m128i _pB8 = _mm_loadu_si16(pB);
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    __m128i _pB = _mm_unpacklo_epi16(_pB16, _mm_setzero_si128());
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA++;
                    pB += 2;
                }
                __m128 _ad1 = _mm_load_ss(pA_descales);
                __m128 _ad = _mm_shuffle_ps(_ad1, _ad1, _MM_SHUFFLE(0, 0, 0, 0));
                __m128 _bd = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pB_descales);
                _fsum = _mm_add_ps(_fsum, _mm_mul_ps(_mm_cvtepi32_ps(_sum), _mm_mul_ps(_ad, _bd)));
                pA_descales += 1;
                pB_descales += 2;
            }

            _mm_storel_pi((__m64*)outptr, _fsum);
            outptr += 2;
#else
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
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                int sum0 = 0;
                int sum1 = 0;
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[2];
                    sum0 += pA[2] * pB[4];
                    sum0 += pA[3] * pB[6];
                    sum1 += pA[0] * pB[1];
                    sum1 += pA[1] * pB[3];
                    sum1 += pA[2] * pB[5];
                    sum1 += pA[3] * pB[7];
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[0] * pB[1];
                    pA++;
                    pB += 2;
                }

                const float ad = pA_descales[0];
                fsum0 += sum0 * ad * pB_descales[0];
                fsum1 += sum1 * ad * pB_descales[1];
                pA_descales += 1;
                pB_descales += 2;
            }

            outptr[0] = fsum0;
            outptr[1] = fsum1;
            outptr += 2;
#endif // __SSE2__
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + (size_t)k;
            const float* pB_descales = pB_descales_panel + (size_t)block_start;
#if __SSE2__
            float fsum;

            if (k == 0)
            {
                fsum = 0.f;
            }
            else
            {
                fsum = outptr[0];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                __m128i _sum = _mm_setzero_si128();
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = _mm_loadu_si32(pA);
                    __m128i _pB = _mm_loadu_si32(pB);
#if __AVXVNNIINT8__
                    _sum = _mm_dpbssd_epi32(_sum, _pB, _pA);
#else  // __AVXVNNIINT8__
                    _sum = _mm_comp_dpbusd_epi32(_sum, _pB, _pA);
#endif // __AVXVNNIINT8__
                    pA += 4;
                    pB += 4;
                }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                if (max_kk0 >= 4)
                {
                    _sum = _mm_sub_epi32(_sum, _mm_loadu_si32(pA));
                    pA += 4;
                }
#endif
#else
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA8 = _mm_loadu_si32(pA);
                    __m128i _pB8 = _mm_loadu_si32(pB);
                    __m128i _pA = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 4;
                    pB += 4;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    __m128i _pA8 = _mm_loadu_si16(pA);
                    __m128i _pB8 = _mm_loadu_si16(pB);
                    __m128i _pA = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _pA, _pB);
                    pA += 2;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA8 = _mm_cvtsi32_si128((unsigned char)pA[0]);
                    __m128i _pB8 = _mm_cvtsi32_si128((unsigned char)pB[0]);
                    __m128i _pA16 = _mm_unpacklo_epi8(_pA8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA8));
                    __m128i _pB16 = _mm_unpacklo_epi8(_pB8, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB8));
                    _sum = _mm_comp_dpwssd_epi32(_sum, _mm_unpacklo_epi16(_pA16, _pA16), _mm_unpacklo_epi16(_pB16, _mm_setzero_si128()));
                    pA++;
                    pB++;
                }
                fsum += _mm_reduce_add_epi32(_sum) * pA_descales[0] * pB_descales[0];
                pA_descales += 1;
                pB_descales++;
            }

            outptr[0] = fsum;
            outptr++;
#else
            float fsum;

            if (k == 0)
            {
                fsum = 0.f;
            }
            else
            {
                fsum = outptr[0];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < tile_K; kk0 += block_size)
            {
                int sum = 0;
                const int max_kk0 = std::min(tile_K - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum += pA[0] * pB[0];
                    sum += pA[1] * pB[1];
                    sum += pA[2] * pB[2];
                    sum += pA[3] * pB[3];
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum += pA[0] * pB[0];
                    pA++;
                    pB++;
                }

                fsum += sum * pA_descales[0] * pB_descales[0];
                pA_descales += 1;
                pB_descales++;
            }

            outptr[0] = fsum;
            outptr++;
#endif // __SSE2__
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        unpack_output_tile_wq_int8_avx512vnni(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        unpack_output_tile_wq_int8_avxvnniint8(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        unpack_output_tile_wq_int8_avxvnni(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        unpack_output_tile_wq_int8_avx2(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
        return;
    }
#endif

    const float* pC = C;
    const float* pp = topT;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;

    // topT microkernel lanes -> m0[n0..nNR-1], ..., mMR-1[n0..nNR-1]
    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;

        __m512 _c = _mm512_setzero_ps();
        __m512i _c_vindex = _mm512_setzero_si512();
        if (pC)
        {
            if (broadcast_type_C == 0)
                _c = _mm512_set1_ps(pC[0]);
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                _c = _mm512_loadu_ps(pC + i + ii);
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j;
                _c_vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _c_vindex = _mm512_mullo_epi32(_c_vindex, _mm512_set1_epi32((int)c_hstep));
            }
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if ((broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
                _c = _mm512_mul_ps(_c, _mm512_set1_ps(beta));
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m512 _f0 = _mm512_loadu_ps(pp + 0);
            __m512 _f1 = _mm512_loadu_ps(pp + 16);
            __m512 _f2 = _mm512_loadu_ps(pp + 32);
            __m512 _f3 = _mm512_loadu_ps(pp + 48);
            __m512 _f4 = _mm512_loadu_ps(pp + 64);
            __m512 _f5 = _mm512_loadu_ps(pp + 80);
            __m512 _f6 = _mm512_loadu_ps(pp + 96);
            __m512 _f7 = _mm512_loadu_ps(pp + 112);
            pp += 128;

            // from
            //      00 11 22 33  44 55 66 77  80 91 a2 b3  c4 d5 e6 f7
            //      01 12 23 30  45 56 67 74  81 92 a3 b0  c5 d6 e7 f4
            //      20 31 02 13  64 75 46 57  a0 b1 82 93  e4 f5 c6 d7
            //      21 32 03 10  65 76 47 54  a1 b2 83 90  e5 f6 c7 d4
            //      04 15 26 37  40 51 62 73  84 95 a6 b7  c0 d1 e2 f3
            //      05 16 27 34  41 52 63 70  85 96 a7 b4  c1 d2 e3 f0
            //      24 35 06 17  60 71 42 53  a4 b5 86 97  e0 f1 c2 d3
            //      25 36 07 14  61 72 43 50  a5 b6 87 94  e1 f2 c3 d0
            //
            // to
            //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
            //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
            //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
            //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
            //      04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
            //      05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
            //      06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
            //      07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
            _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
            __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
            __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
            __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
            __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
            __m512 _tmp4 = _mm512_unpacklo_ps(_f4, _f7);
            __m512 _tmp5 = _mm512_unpackhi_ps(_f4, _f7);
            __m512 _tmp6 = _mm512_unpacklo_ps(_f6, _f5);
            __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f5);
            _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
            _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
            _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
            _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
            _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
            _f5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
            _f6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
            _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
            _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
            _tmp0 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(0, 1, 1, 0));
            _tmp1 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
            _tmp2 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
            _tmp3 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
            _tmp4 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
            _tmp5 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
            _tmp6 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
            _tmp7 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));
            _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
            _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
            _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
            _f3 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
            _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 3, 1, 3));
            _f5 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
            _f6 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 3, 1, 3));
            _f7 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c);
                    _f1 = _mm512_add_ps(_f1, _c);
                    _f2 = _mm512_add_ps(_f2, _c);
                    _f3 = _mm512_add_ps(_f3, _c);
                    _f4 = _mm512_add_ps(_f4, _c);
                    _f5 = _mm512_add_ps(_f5, _c);
                    _f6 = _mm512_add_ps(_f6, _c);
                    _f7 = _mm512_add_ps(_f7, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c0 = _mm512_i32gather_ps(_c_vindex, pC, sizeof(float));
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_i32gather_ps(_c_vindex, pC + 1, sizeof(float));
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    __m512 _c2 = _mm512_i32gather_ps(_c_vindex, pC + 2, sizeof(float));
                    if (beta != 1.f)
                        _c2 = _mm512_mul_ps(_c2, _mm512_set1_ps(beta));
                    _f2 = _mm512_add_ps(_f2, _c2);
                    __m512 _c3 = _mm512_i32gather_ps(_c_vindex, pC + 3, sizeof(float));
                    if (beta != 1.f)
                        _c3 = _mm512_mul_ps(_c3, _mm512_set1_ps(beta));
                    _f3 = _mm512_add_ps(_f3, _c3);
                    __m512 _c4 = _mm512_i32gather_ps(_c_vindex, pC + 4, sizeof(float));
                    if (beta != 1.f)
                        _c4 = _mm512_mul_ps(_c4, _mm512_set1_ps(beta));
                    _f4 = _mm512_add_ps(_f4, _c4);
                    __m512 _c5 = _mm512_i32gather_ps(_c_vindex, pC + 5, sizeof(float));
                    if (beta != 1.f)
                        _c5 = _mm512_mul_ps(_c5, _mm512_set1_ps(beta));
                    _f5 = _mm512_add_ps(_f5, _c5);
                    __m512 _c6 = _mm512_i32gather_ps(_c_vindex, pC + 6, sizeof(float));
                    if (beta != 1.f)
                        _c6 = _mm512_mul_ps(_c6, _mm512_set1_ps(beta));
                    _f6 = _mm512_add_ps(_f6, _c6);
                    __m512 _c7 = _mm512_i32gather_ps(_c_vindex, pC + 7, sizeof(float));
                    if (beta != 1.f)
                        _c7 = _mm512_mul_ps(_c7, _mm512_set1_ps(beta));
                    _f7 = _mm512_add_ps(_f7, _c7);
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _c0 = _mm512_set1_ps(pC[0]);
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_set1_ps(pC[1]);
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    __m512 _c2 = _mm512_set1_ps(pC[2]);
                    if (beta != 1.f)
                        _c2 = _mm512_mul_ps(_c2, _mm512_set1_ps(beta));
                    _f2 = _mm512_add_ps(_f2, _c2);
                    __m512 _c3 = _mm512_set1_ps(pC[3]);
                    if (beta != 1.f)
                        _c3 = _mm512_mul_ps(_c3, _mm512_set1_ps(beta));
                    _f3 = _mm512_add_ps(_f3, _c3);
                    __m512 _c4 = _mm512_set1_ps(pC[4]);
                    if (beta != 1.f)
                        _c4 = _mm512_mul_ps(_c4, _mm512_set1_ps(beta));
                    _f4 = _mm512_add_ps(_f4, _c4);
                    __m512 _c5 = _mm512_set1_ps(pC[5]);
                    if (beta != 1.f)
                        _c5 = _mm512_mul_ps(_c5, _mm512_set1_ps(beta));
                    _f5 = _mm512_add_ps(_f5, _c5);
                    __m512 _c6 = _mm512_set1_ps(pC[6]);
                    if (beta != 1.f)
                        _c6 = _mm512_mul_ps(_c6, _mm512_set1_ps(beta));
                    _f6 = _mm512_add_ps(_f6, _c6);
                    __m512 _c7 = _mm512_set1_ps(pC[7]);
                    if (beta != 1.f)
                        _c7 = _mm512_mul_ps(_c7, _mm512_set1_ps(beta));
                    _f7 = _mm512_add_ps(_f7, _c7);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
                _f4 = _mm512_mul_ps(_f4, _alpha);
                _f5 = _mm512_mul_ps(_f5, _alpha);
                _f6 = _mm512_mul_ps(_f6, _alpha);
                _f7 = _mm512_mul_ps(_f7, _alpha);
            }
            transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
            _mm256_storeu_ps(p0, _mm512_castps512_ps256(_f0));
            _mm256_storeu_ps(p0 + out_hstep, _mm512_extractf32x8_ps(_f0, 1));
            _mm256_storeu_ps(p0 + out_hstep * 2, _mm512_castps512_ps256(_f1));
            _mm256_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x8_ps(_f1, 1));
            _mm256_storeu_ps(p0 + out_hstep * 4, _mm512_castps512_ps256(_f2));
            _mm256_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x8_ps(_f2, 1));
            _mm256_storeu_ps(p0 + out_hstep * 6, _mm512_castps512_ps256(_f3));
            _mm256_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x8_ps(_f3, 1));
            _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_castps512_ps256(_f4));
            _mm256_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x8_ps(_f4, 1));
            _mm256_storeu_ps(p0 + out_hstep * 10, _mm512_castps512_ps256(_f5));
            _mm256_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x8_ps(_f5, 1));
            _mm256_storeu_ps(p0 + out_hstep * 12, _mm512_castps512_ps256(_f6));
            _mm256_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x8_ps(_f6, 1));
            _mm256_storeu_ps(p0 + out_hstep * 14, _mm512_castps512_ps256(_f7));
            _mm256_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x8_ps(_f7, 1));
            p0 += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512 _f0 = _mm512_loadu_ps(pp + 0);
            __m512 _f1 = _mm512_loadu_ps(pp + 16);
            __m512 _f2 = _mm512_loadu_ps(pp + 32);
            __m512 _f3 = _mm512_loadu_ps(pp + 48);
            pp += 64;
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
            __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
            __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
            __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
            _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
            _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
            _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
            _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c);
                    _f1 = _mm512_add_ps(_f1, _c);
                    _f2 = _mm512_add_ps(_f2, _c);
                    _f3 = _mm512_add_ps(_f3, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c0 = _mm512_i32gather_ps(_c_vindex, pC, sizeof(float));
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_i32gather_ps(_c_vindex, pC + 1, sizeof(float));
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    __m512 _c2 = _mm512_i32gather_ps(_c_vindex, pC + 2, sizeof(float));
                    if (beta != 1.f)
                        _c2 = _mm512_mul_ps(_c2, _mm512_set1_ps(beta));
                    _f2 = _mm512_add_ps(_f2, _c2);
                    __m512 _c3 = _mm512_i32gather_ps(_c_vindex, pC + 3, sizeof(float));
                    if (beta != 1.f)
                        _c3 = _mm512_mul_ps(_c3, _mm512_set1_ps(beta));
                    _f3 = _mm512_add_ps(_f3, _c3);
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _c0 = _mm512_set1_ps(pC[0]);
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_set1_ps(pC[1]);
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    __m512 _c2 = _mm512_set1_ps(pC[2]);
                    if (beta != 1.f)
                        _c2 = _mm512_mul_ps(_c2, _mm512_set1_ps(beta));
                    _f2 = _mm512_add_ps(_f2, _c2);
                    __m512 _c3 = _mm512_set1_ps(pC[3]);
                    if (beta != 1.f)
                        _c3 = _mm512_mul_ps(_c3, _mm512_set1_ps(beta));
                    _f3 = _mm512_add_ps(_f3, _c3);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
            }
            transpose16x4_ps(_f0, _f1, _f2, _f3);
            _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
            _mm_storeu_ps(p0 + out_hstep, _mm512_extractf32x4_ps(_f0, 1));
            _mm_storeu_ps(p0 + out_hstep * 2, _mm512_extractf32x4_ps(_f0, 2));
            _mm_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x4_ps(_f0, 3));
            _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f1, 0));
            _mm_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x4_ps(_f1, 1));
            _mm_storeu_ps(p0 + out_hstep * 6, _mm512_extractf32x4_ps(_f1, 2));
            _mm_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x4_ps(_f1, 3));
            _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f2, 0));
            _mm_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x4_ps(_f2, 1));
            _mm_storeu_ps(p0 + out_hstep * 10, _mm512_extractf32x4_ps(_f2, 2));
            _mm_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x4_ps(_f2, 3));
            _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f3, 0));
            _mm_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x4_ps(_f3, 1));
            _mm_storeu_ps(p0 + out_hstep * 14, _mm512_extractf32x4_ps(_f3, 2));
            _mm_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x4_ps(_f3, 3));
            p0 += 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512 _f0 = _mm512_loadu_ps(pp + 0);
            __m512 _f1 = _mm512_loadu_ps(pp + 16);
            pp += 32;
            __m512 _tmp0 = _mm512_permute_ps(_f0, _MM_SHUFFLE(3, 1, 2, 0));
            __m512 _tmp1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(0, 2, 3, 1));
            _f0 = _mm512_unpacklo_ps(_tmp0, _tmp1);
            _f1 = _mm512_unpackhi_ps(_tmp0, _tmp1);
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c);
                    _f1 = _mm512_add_ps(_f1, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c0 = _mm512_i32gather_ps(_c_vindex, pC, sizeof(float));
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_i32gather_ps(_c_vindex, pC + 1, sizeof(float));
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _c0 = _mm512_set1_ps(pC[0]);
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_set1_ps(pC[1]);
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
            }
            transpose16x2_ps(_f0, _f1);
            {
                __m128 _r = _mm512_extractf32x4_ps(_f0, 0);
                _mm_storel_pi((__m64*)(p0), _r);
                _mm_storeh_pi((__m64*)(p0 + out_hstep), _r);
            }
            {
                __m128 _r = _mm512_extractf32x4_ps(_f0, 1);
                _mm_storel_pi((__m64*)(p0 + out_hstep * 2), _r);
                _mm_storeh_pi((__m64*)(p0 + out_hstep * 3), _r);
            }
            {
                __m128 _r = _mm512_extractf32x4_ps(_f0, 2);
                _mm_storel_pi((__m64*)(p0 + out_hstep * 4), _r);
                _mm_storeh_pi((__m64*)(p0 + out_hstep * 5), _r);
            }
            {
                __m128 _r = _mm512_extractf32x4_ps(_f0, 3);
                _mm_storel_pi((__m64*)(p0 + out_hstep * 6), _r);
                _mm_storeh_pi((__m64*)(p0 + out_hstep * 7), _r);
            }
            {
                __m128 _r = _mm512_extractf32x4_ps(_f1, 0);
                _mm_storel_pi((__m64*)(p0 + out_hstep * 8), _r);
                _mm_storeh_pi((__m64*)(p0 + out_hstep * 9), _r);
            }
            {
                __m128 _r = _mm512_extractf32x4_ps(_f1, 1);
                _mm_storel_pi((__m64*)(p0 + out_hstep * 10), _r);
                _mm_storeh_pi((__m64*)(p0 + out_hstep * 11), _r);
            }
            {
                __m128 _r = _mm512_extractf32x4_ps(_f1, 2);
                _mm_storel_pi((__m64*)(p0 + out_hstep * 12), _r);
                _mm_storeh_pi((__m64*)(p0 + out_hstep * 13), _r);
            }
            {
                __m128 _r = _mm512_extractf32x4_ps(_f1, 3);
                _mm_storel_pi((__m64*)(p0 + out_hstep * 14), _r);
                _mm_storeh_pi((__m64*)(p0 + out_hstep * 15), _r);
            }
            p0 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            __m512 _f0 = _mm512_loadu_ps(pp + 0);
            pp += 16;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c0 = _mm512_i32gather_ps(_c_vindex, pC, sizeof(float));
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _c0 = _mm512_set1_ps(pC[0]);
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
            }
            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32((int)out_hstep));
            _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
            p0++;
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    const float* pp1 = pp + max_jj * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;

        float c0 = 0.f;
        float c1 = c0;
        float c2 = c0;
        float c3 = c0;
        float c4 = c0;
        float c5 = c0;
        float c6 = c0;
        float c7 = c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                c1 = c0;
                c2 = c0;
                c3 = c0;
                c4 = c0;
                c5 = c0;
                c6 = c0;
                c7 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                c1 = pC[i + ii + 1];
                c2 = pC[i + ii + 2];
                c3 = pC[i + ii + 3];
                c4 = pC[i + ii + 4];
                c5 = pC[i + ii + 5];
                c6 = pC[i + ii + 6];
                c7 = pC[i + ii + 7];
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if (broadcast_type_C == 0 && beta != 1.f)
                c0 *= beta;
            if ((broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
            {
                c0 *= beta;
                c1 *= beta;
                c2 *= beta;
                c3 *= beta;
                c4 *= beta;
                c5 *= beta;
                c6 *= beta;
                c7 *= beta;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = _mm256_loadu_ps(pp + 0);
            __m256 _f1 = _mm256_loadu_ps(pp + 8);
            __m256 _f2 = _mm256_loadu_ps(pp + 16);
            __m256 _f3 = _mm256_loadu_ps(pp + 24);
            __m256 _f4 = _mm256_loadu_ps(pp + 32);
            __m256 _f5 = _mm256_loadu_ps(pp + 40);
            __m256 _f6 = _mm256_loadu_ps(pp + 48);
            __m256 _f7 = _mm256_loadu_ps(pp + 56);
            pp += 64;

            // from
            //      00 11 22 33 44 55 66 77
            //      01 12 23 30 45 56 67 74
            //      20 31 02 13 64 75 46 57
            //      21 32 03 10 65 76 47 54
            //      04 15 26 37 40 51 62 73
            //      05 16 27 34 41 52 63 70
            //      24 35 06 17 60 71 42 53
            //      25 36 07 14 61 72 43 50

            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            //      02 12 22 32 42 52 62 72
            //      03 13 23 33 43 53 63 73
            //      04 14 24 34 44 54 64 74
            //      05 15 25 35 45 55 65 75
            //      06 16 26 36 46 56 66 76
            //      07 17 27 37 47 57 67 77
            {
                __m256 _tmp0 = _f0;
                __m256 _tmp1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp2 = _f2;
                __m256 _tmp3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _f4;
                __m256 _tmp5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp6 = _f6;
                __m256 _tmp7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
                _f0 = _mm256_unpacklo_ps(_tmp0, _tmp3);
                _f1 = _mm256_unpackhi_ps(_tmp0, _tmp3);
                _f2 = _mm256_unpacklo_ps(_tmp2, _tmp1);
                _f3 = _mm256_unpackhi_ps(_tmp2, _tmp1);
                _f4 = _mm256_unpacklo_ps(_tmp4, _tmp7);
                _f5 = _mm256_unpackhi_ps(_tmp4, _tmp7);
                _f6 = _mm256_unpacklo_ps(_tmp6, _tmp5);
                _f7 = _mm256_unpackhi_ps(_tmp6, _tmp5);
                _tmp0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f0), _mm256_castps_pd(_f2)));
                _tmp1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f0), _mm256_castps_pd(_f2)));
                _tmp2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f3), _mm256_castps_pd(_f1)));
                _tmp3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f3), _mm256_castps_pd(_f1)));
                _tmp4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f4), _mm256_castps_pd(_f6)));
                _tmp5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f4), _mm256_castps_pd(_f6)));
                _tmp6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f7), _mm256_castps_pd(_f5)));
                _tmp7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f7), _mm256_castps_pd(_f5)));
                _tmp1 = _mm256_shuffle_ps(_tmp1, _tmp1, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp3 = _mm256_shuffle_ps(_tmp3, _tmp3, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp5 = _mm256_shuffle_ps(_tmp5, _tmp5, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp7 = _mm256_shuffle_ps(_tmp7, _tmp7, _MM_SHUFFLE(2, 1, 0, 3));
                _f0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 0));
                _f1 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 0));
                _f2 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 0));
                _f3 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 0));
                _f4 = _mm256_permute2f128_ps(_tmp4, _tmp0, _MM_SHUFFLE(0, 3, 0, 0));
                _f5 = _mm256_permute2f128_ps(_tmp5, _tmp1, _MM_SHUFFLE(0, 3, 0, 0));
                _f6 = _mm256_permute2f128_ps(_tmp6, _tmp2, _MM_SHUFFLE(0, 3, 0, 0));
                _f7 = _mm256_permute2f128_ps(_tmp7, _tmp3, _MM_SHUFFLE(0, 3, 0, 0));
            }
            transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = _mm256_set1_ps(c0);
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    _f2 = _mm256_add_ps(_f2, _c);
                    _f3 = _mm256_add_ps(_f3, _c);
                    _f4 = _mm256_add_ps(_f4, _c);
                    _f5 = _mm256_add_ps(_f5, _c);
                    _f6 = _mm256_add_ps(_f6, _c);
                    _f7 = _mm256_add_ps(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(c0));
                    _f1 = _mm256_add_ps(_f1, _mm256_set1_ps(c1));
                    _f2 = _mm256_add_ps(_f2, _mm256_set1_ps(c2));
                    _f3 = _mm256_add_ps(_f3, _mm256_set1_ps(c3));
                    _f4 = _mm256_add_ps(_f4, _mm256_set1_ps(c4));
                    _f5 = _mm256_add_ps(_f5, _mm256_set1_ps(c5));
                    _f6 = _mm256_add_ps(_f6, _mm256_set1_ps(c6));
                    _f7 = _mm256_add_ps(_f7, _mm256_set1_ps(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = _mm256_loadu_ps(pC);
                    __m256 _c1 = _mm256_loadu_ps(pC + c_hstep);
                    __m256 _c2 = _mm256_loadu_ps(pC + c_hstep * 2);
                    __m256 _c3 = _mm256_loadu_ps(pC + c_hstep * 3);
                    __m256 _c4 = _mm256_loadu_ps(pC + c_hstep * 4);
                    __m256 _c5 = _mm256_loadu_ps(pC + c_hstep * 5);
                    __m256 _c6 = _mm256_loadu_ps(pC + c_hstep * 6);
                    __m256 _c7 = _mm256_loadu_ps(pC + c_hstep * 7);
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                        _f2 = _mm256_add_ps(_f2, _c2);
                        _f3 = _mm256_add_ps(_f3, _c3);
                        _f4 = _mm256_add_ps(_f4, _c4);
                        _f5 = _mm256_add_ps(_f5, _c5);
                        _f6 = _mm256_add_ps(_f6, _c6);
                        _f7 = _mm256_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
#if __FMA__
                        _f0 = _mm256_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm256_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm256_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm256_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm256_fmadd_ps(_c7, _beta, _f7);
#else
                        _f0 = _mm256_add_ps(_f0, _mm256_mul_ps(_c0, _beta));
                        _f1 = _mm256_add_ps(_f1, _mm256_mul_ps(_c1, _beta));
                        _f2 = _mm256_add_ps(_f2, _mm256_mul_ps(_c2, _beta));
                        _f3 = _mm256_add_ps(_f3, _mm256_mul_ps(_c3, _beta));
                        _f4 = _mm256_add_ps(_f4, _mm256_mul_ps(_c4, _beta));
                        _f5 = _mm256_add_ps(_f5, _mm256_mul_ps(_c5, _beta));
                        _f6 = _mm256_add_ps(_f6, _mm256_mul_ps(_c6, _beta));
                        _f7 = _mm256_add_ps(_f7, _mm256_mul_ps(_c7, _beta));
#endif
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = _mm256_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm256_mul_ps(_c, _mm256_set1_ps(beta));
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    _f2 = _mm256_add_ps(_f2, _c);
                    _f3 = _mm256_add_ps(_f3, _c);
                    _f4 = _mm256_add_ps(_f4, _c);
                    _f5 = _mm256_add_ps(_f5, _c);
                    _f6 = _mm256_add_ps(_f6, _c);
                    _f7 = _mm256_add_ps(_f7, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
                _f4 = _mm256_mul_ps(_f4, _alpha);
                _f5 = _mm256_mul_ps(_f5, _alpha);
                _f6 = _mm256_mul_ps(_f6, _alpha);
                _f7 = _mm256_mul_ps(_f7, _alpha);
            }
            _mm256_storeu_ps(p0, _f0);
            _mm256_storeu_ps(p0 + out_hstep, _f1);
            _mm256_storeu_ps(p0 + out_hstep * 2, _f2);
            _mm256_storeu_ps(p0 + out_hstep * 3, _f3);
            _mm256_storeu_ps(p0 + out_hstep * 4, _f4);
            _mm256_storeu_ps(p0 + out_hstep * 5, _f5);
            _mm256_storeu_ps(p0 + out_hstep * 6, _f6);
            _mm256_storeu_ps(p0 + out_hstep * 7, _f7);
            p0 += 8;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
#if __AVX2__
            __m256 _t0 = _mm256_loadu_ps(pp + 0);
            __m256 _t1 = _mm256_loadu_ps(pp + 8);
            __m256 _t2 = _mm256_loadu_ps(pp + 16);
            __m256 _t3 = _mm256_loadu_ps(pp + 24);
#else
            __m256 _t0 = combine4x2_ps(_mm_loadu_ps(pp + 0), _mm_loadu_ps(pp1 + 0));
            __m256 _t1 = combine4x2_ps(_mm_loadu_ps(pp + 4), _mm_loadu_ps(pp1 + 4));
            __m256 _t2 = combine4x2_ps(_mm_loadu_ps(pp + 8), _mm_loadu_ps(pp1 + 8));
            __m256 _t3 = combine4x2_ps(_mm_loadu_ps(pp + 12), _mm_loadu_ps(pp1 + 12));
#endif
#if __AVX2__
            pp += 32;
#else
            pp += 16;
            pp1 += 16;
#endif
            _t1 = _mm256_shuffle_ps(_t1, _t1, _MM_SHUFFLE(2, 1, 0, 3));
            _t3 = _mm256_shuffle_ps(_t3, _t3, _MM_SHUFFLE(2, 1, 0, 3));
            __m256 _tmp0 = _mm256_unpacklo_ps(_t0, _t3);
            __m256 _tmp1 = _mm256_unpackhi_ps(_t0, _t3);
            __m256 _tmp2 = _mm256_unpacklo_ps(_t2, _t1);
            __m256 _tmp3 = _mm256_unpackhi_ps(_t2, _t1);
            _t0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
            _t1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
            _t2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
            _t3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
            _t1 = _mm256_shuffle_ps(_t1, _t1, _MM_SHUFFLE(2, 1, 0, 3));
            _t3 = _mm256_shuffle_ps(_t3, _t3, _MM_SHUFFLE(2, 1, 0, 3));
            transpose8x4_ps(_t0, _t1, _t2, _t3);
            __m128 _f0 = _mm256_extractf128_ps(_t0, 0);
            __m128 _f1 = _mm256_extractf128_ps(_t0, 1);
            __m128 _f2 = _mm256_extractf128_ps(_t1, 0);
            __m128 _f3 = _mm256_extractf128_ps(_t1, 1);
            __m128 _f4 = _mm256_extractf128_ps(_t2, 0);
            __m128 _f5 = _mm256_extractf128_ps(_t2, 1);
            __m128 _f6 = _mm256_extractf128_ps(_t3, 0);
            __m128 _f7 = _mm256_extractf128_ps(_t3, 1);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    _f4 = _mm_add_ps(_f4, _c);
                    _f5 = _mm_add_ps(_f5, _c);
                    _f6 = _mm_add_ps(_f6, _c);
                    _f7 = _mm_add_ps(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                    _f2 = _mm_add_ps(_f2, _mm_set1_ps(c2));
                    _f3 = _mm_add_ps(_f3, _mm_set1_ps(c3));
                    _f4 = _mm_add_ps(_f4, _mm_set1_ps(c4));
                    _f5 = _mm_add_ps(_f5, _mm_set1_ps(c5));
                    _f6 = _mm_add_ps(_f6, _mm_set1_ps(c6));
                    _f7 = _mm_add_ps(_f7, _mm_set1_ps(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                    __m128 _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                    __m128 _c4 = _mm_loadu_ps(pC + c_hstep * 4);
                    __m128 _c5 = _mm_loadu_ps(pC + c_hstep * 5);
                    __m128 _c6 = _mm_loadu_ps(pC + c_hstep * 6);
                    __m128 _c7 = _mm_loadu_ps(pC + c_hstep * 7);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                        _f4 = _mm_add_ps(_f4, _c4);
                        _f5 = _mm_add_ps(_f5, _c5);
                        _f6 = _mm_add_ps(_f6, _c6);
                        _f7 = _mm_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm_fmadd_ps(_c7, _beta, _f7);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
                        _f2 = _mm_add_ps(_f2, _mm_mul_ps(_c2, _beta));
                        _f3 = _mm_add_ps(_f3, _mm_mul_ps(_c3, _beta));
                        _f4 = _mm_add_ps(_f4, _mm_mul_ps(_c4, _beta));
                        _f5 = _mm_add_ps(_f5, _mm_mul_ps(_c5, _beta));
                        _f6 = _mm_add_ps(_f6, _mm_mul_ps(_c6, _beta));
                        _f7 = _mm_add_ps(_f7, _mm_mul_ps(_c7, _beta));
#endif
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    _f4 = _mm_add_ps(_f4, _c);
                    _f5 = _mm_add_ps(_f5, _c);
                    _f6 = _mm_add_ps(_f6, _c);
                    _f7 = _mm_add_ps(_f7, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
                _f4 = _mm_mul_ps(_f4, _alpha);
                _f5 = _mm_mul_ps(_f5, _alpha);
                _f6 = _mm_mul_ps(_f6, _alpha);
                _f7 = _mm_mul_ps(_f7, _alpha);
            }
            _mm_storeu_ps(p0, _f0);
            _mm_storeu_ps(p0 + out_hstep, _f1);
            _mm_storeu_ps(p0 + out_hstep * 2, _f2);
            _mm_storeu_ps(p0 + out_hstep * 3, _f3);
            _mm_storeu_ps(p0 + out_hstep * 4, _f4);
            _mm_storeu_ps(p0 + out_hstep * 5, _f5);
            _mm_storeu_ps(p0 + out_hstep * 6, _f6);
            _mm_storeu_ps(p0 + out_hstep * 7, _f7);
            p0 += 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __AVX2__
            __m256 _t0 = _mm256_loadu_ps(pp);
            __m256 _t1 = _mm256_loadu_ps(pp + 8);
#else
            __m256 _t0 = combine4x2_ps(_mm_loadu_ps(pp), _mm_loadu_ps(pp1));
            __m256 _t1 = combine4x2_ps(_mm_loadu_ps(pp + 4), _mm_loadu_ps(pp1 + 4));
#endif
#if __AVX2__
            pp += 16;
#else
            pp += 8;
            pp1 += 8;
#endif
            __m256 _tmp0 = _mm256_shuffle_ps(_t0, _t0, _MM_SHUFFLE(3, 1, 2, 0));
            __m256 _tmp1 = _mm256_shuffle_ps(_t1, _t1, _MM_SHUFFLE(0, 2, 3, 1));
            _t0 = _mm256_unpacklo_ps(_tmp0, _tmp1);
            _t1 = _mm256_unpackhi_ps(_tmp0, _tmp1);
            _t1 = _mm256_shuffle_ps(_t1, _t1, _MM_SHUFFLE(2, 1, 0, 3));
            transpose8x2_ps(_t0, _t1);
            __m128 _r01 = _mm256_extractf128_ps(_t0, 0);
            __m128 _r23 = _mm256_extractf128_ps(_t0, 1);
            __m128 _r45 = _mm256_extractf128_ps(_t1, 0);
            __m128 _r67 = _mm256_extractf128_ps(_t1, 1);
            __m128 _f0 = _r01;
            __m128 _f1 = _mm_movehl_ps(_r01, _r01);
            __m128 _f2 = _r23;
            __m128 _f3 = _mm_movehl_ps(_r23, _r23);
            __m128 _f4 = _r45;
            __m128 _f5 = _mm_movehl_ps(_r45, _r45);
            __m128 _f6 = _r67;
            __m128 _f7 = _mm_movehl_ps(_r67, _r67);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    _f4 = _mm_add_ps(_f4, _c);
                    _f5 = _mm_add_ps(_f5, _c);
                    _f6 = _mm_add_ps(_f6, _c);
                    _f7 = _mm_add_ps(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                    _f2 = _mm_add_ps(_f2, _mm_set1_ps(c2));
                    _f3 = _mm_add_ps(_f3, _mm_set1_ps(c3));
                    _f4 = _mm_add_ps(_f4, _mm_set1_ps(c4));
                    _f5 = _mm_add_ps(_f5, _mm_set1_ps(c5));
                    _f6 = _mm_add_ps(_f6, _mm_set1_ps(c6));
                    _f7 = _mm_add_ps(_f7, _mm_set1_ps(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    __m128 _c1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep));
                    __m128 _c2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 2));
                    __m128 _c3 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 3));
                    __m128 _c4 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 4));
                    __m128 _c5 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 5));
                    __m128 _c6 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 6));
                    __m128 _c7 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 7));
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                        _f4 = _mm_add_ps(_f4, _c4);
                        _f5 = _mm_add_ps(_f5, _c5);
                        _f6 = _mm_add_ps(_f6, _c6);
                        _f7 = _mm_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm_fmadd_ps(_c7, _beta, _f7);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
                        _f2 = _mm_add_ps(_f2, _mm_mul_ps(_c2, _beta));
                        _f3 = _mm_add_ps(_f3, _mm_mul_ps(_c3, _beta));
                        _f4 = _mm_add_ps(_f4, _mm_mul_ps(_c4, _beta));
                        _f5 = _mm_add_ps(_f5, _mm_mul_ps(_c5, _beta));
                        _f6 = _mm_add_ps(_f6, _mm_mul_ps(_c6, _beta));
                        _f7 = _mm_add_ps(_f7, _mm_mul_ps(_c7, _beta));
#endif
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    _f4 = _mm_add_ps(_f4, _c);
                    _f5 = _mm_add_ps(_f5, _c);
                    _f6 = _mm_add_ps(_f6, _c);
                    _f7 = _mm_add_ps(_f7, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
                _f4 = _mm_mul_ps(_f4, _alpha);
                _f5 = _mm_mul_ps(_f5, _alpha);
                _f6 = _mm_mul_ps(_f6, _alpha);
                _f7 = _mm_mul_ps(_f7, _alpha);
            }
            _mm_storel_pi((__m64*)p0, _f0);
            _mm_storel_pi((__m64*)(p0 + out_hstep), _f1);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 2), _f2);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 3), _f3);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 4), _f4);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 5), _f5);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 6), _f6);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 7), _f7);
            p0 += 2;
        }
        for (; jj < max_jj; jj++)
        {
#if __AVX2__
            __m256 _f0 = _mm256_loadu_ps(pp);
#else
            __m256 _f0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(pp)), _mm_loadu_ps(pp1), 1);
#endif
#if __AVX2__
            pp += 8;
#else
            pp += 4;
            pp1 += 4;
#endif
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _mm256_setr_ps(c0, c1, c2, c3, c4, c5, c6, c7));
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m256 _c;
                    if (broadcast_type_C == 3)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32((int)c_hstep));
                        _c = _mm256_i32gather_ps(pC, _vindex, sizeof(float));
#else
                        _c = _mm256_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3], pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]);
#endif
                    }
                    if (broadcast_type_C == 4)
                        _c = _mm256_set1_ps(pC[0]);
                    if (beta == 1.f)
                        _f0 = _mm256_add_ps(_f0, _c);
                    else
                        _f0 = _mm256_comp_fmadd_ps(_c, _mm256_set1_ps(beta), _f0);
                    pC++;
                }
            }
            if (alpha != 1.f)
                _f0 = _mm256_mul_ps(_f0, _mm256_set1_ps(alpha));
#if __AVX512F__
            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32((int)out_hstep));
            _mm256_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
#else
#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float sum0[8];
            _mm256_store_ps(sum0, _f0);
            p0[0] = sum0[0];
            p0[out_hstep] = sum0[1];
            p0[out_hstep * 2] = sum0[2];
            p0[out_hstep * 3] = sum0[3];
            p0[out_hstep * 4] = sum0[4];
            p0[out_hstep * 5] = sum0[5];
            p0[out_hstep * 6] = sum0[6];
            p0[out_hstep * 7] = sum0[7];
#endif
            p0++;
        }
#if !__AVX2__
        pp = pp1;
        pp1 = pp + max_jj * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;

        float c0 = 0.f;
        float c1 = c0;
        float c2 = c0;
        float c3 = c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                c1 = c0;
                c2 = c0;
                c3 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                c1 = pC[i + ii + 1];
                c2 = pC[i + ii + 2];
                c3 = pC[i + ii + 3];
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if (broadcast_type_C == 0 && beta != 1.f)
                c0 *= beta;
            if ((broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
            {
                c0 *= beta;
                c1 *= beta;
                c2 *= beta;
                c3 *= beta;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = _mm256_loadu_ps(pp + 0);
            __m256 _f1 = _mm256_loadu_ps(pp + 8);
            __m256 _f2 = _mm256_loadu_ps(pp + 16);
            __m256 _f3 = _mm256_loadu_ps(pp + 24);
            pp += 32;
            __m128 _f00 = _mm256_castps256_ps128(_f0);
            __m128 _f01 = _mm256_castps256_ps128(_f1);
            __m128 _f02 = _mm256_castps256_ps128(_f2);
            __m128 _f03 = _mm256_castps256_ps128(_f3);
            __m128 _f10 = _mm256_extractf128_ps(_f0, 1);
            __m128 _f11 = _mm256_extractf128_ps(_f1, 1);
            __m128 _f12 = _mm256_extractf128_ps(_f2, 1);
            __m128 _f13 = _mm256_extractf128_ps(_f3, 1);
            {
                _f01 = _mm_shuffle_ps(_f01, _f01, _MM_SHUFFLE(2, 1, 0, 3));
                _f03 = _mm_shuffle_ps(_f03, _f03, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f00, _f03);
                __m128 _tmp1 = _mm_unpackhi_ps(_f00, _f03);
                __m128 _tmp2 = _mm_unpacklo_ps(_f02, _f01);
                __m128 _tmp3 = _mm_unpackhi_ps(_f02, _f01);
                _f00 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f01 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f02 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f03 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f01 = _mm_shuffle_ps(_f01, _f01, _MM_SHUFFLE(2, 1, 0, 3));
                _f03 = _mm_shuffle_ps(_f03, _f03, _MM_SHUFFLE(2, 1, 0, 3));
                _MM_TRANSPOSE4_PS(_f00, _f01, _f02, _f03);
            }
            {
                _f11 = _mm_shuffle_ps(_f11, _f11, _MM_SHUFFLE(2, 1, 0, 3));
                _f13 = _mm_shuffle_ps(_f13, _f13, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f10, _f13);
                __m128 _tmp1 = _mm_unpackhi_ps(_f10, _f13);
                __m128 _tmp2 = _mm_unpacklo_ps(_f12, _f11);
                __m128 _tmp3 = _mm_unpackhi_ps(_f12, _f11);
                _f10 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f11 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f11 = _mm_shuffle_ps(_f11, _f11, _MM_SHUFFLE(2, 1, 0, 3));
                _f13 = _mm_shuffle_ps(_f13, _f13, _MM_SHUFFLE(2, 1, 0, 3));
                _MM_TRANSPOSE4_PS(_f10, _f11, _f12, _f13);
            }
            _f0 = combine4x2_ps(_f00, _f10);
            _f1 = combine4x2_ps(_f01, _f11);
            _f2 = combine4x2_ps(_f02, _f12);
            _f3 = combine4x2_ps(_f03, _f13);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = _mm256_set1_ps(c0);
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    _f2 = _mm256_add_ps(_f2, _c);
                    _f3 = _mm256_add_ps(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(c0));
                    _f1 = _mm256_add_ps(_f1, _mm256_set1_ps(c1));
                    _f2 = _mm256_add_ps(_f2, _mm256_set1_ps(c2));
                    _f3 = _mm256_add_ps(_f3, _mm256_set1_ps(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = _mm256_loadu_ps(pC);
                    __m256 _c1 = _mm256_loadu_ps(pC + c_hstep);
                    __m256 _c2 = _mm256_loadu_ps(pC + c_hstep * 2);
                    __m256 _c3 = _mm256_loadu_ps(pC + c_hstep * 3);
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                        _f2 = _mm256_add_ps(_f2, _c2);
                        _f3 = _mm256_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
#if __FMA__
                        _f0 = _mm256_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_fmadd_ps(_c3, _beta, _f3);
#else
                        _f0 = _mm256_add_ps(_f0, _mm256_mul_ps(_c0, _beta));
                        _f1 = _mm256_add_ps(_f1, _mm256_mul_ps(_c1, _beta));
                        _f2 = _mm256_add_ps(_f2, _mm256_mul_ps(_c2, _beta));
                        _f3 = _mm256_add_ps(_f3, _mm256_mul_ps(_c3, _beta));
#endif
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = _mm256_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm256_mul_ps(_c, _mm256_set1_ps(beta));
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    _f2 = _mm256_add_ps(_f2, _c);
                    _f3 = _mm256_add_ps(_f3, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
            }
            _mm256_storeu_ps(p0, _f0);
            _mm256_storeu_ps(p0 + out_hstep, _f1);
            _mm256_storeu_ps(p0 + out_hstep * 2, _f2);
            _mm256_storeu_ps(p0 + out_hstep * 3, _f3);
            p0 += 8;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_loadu_ps(pp + 0);
            __m128 _f1 = _mm_loadu_ps(pp + 4);
            __m128 _f2 = _mm_loadu_ps(pp + 8);
            __m128 _f3 = _mm_loadu_ps(pp + 12);
            pp += 16;
            {
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f3);
                __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f3);
                __m128 _tmp2 = _mm_unpacklo_ps(_f2, _f1);
                __m128 _tmp3 = _mm_unpackhi_ps(_f2, _f1);
                _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
            }
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                    _f2 = _mm_add_ps(_f2, _mm_set1_ps(c2));
                    _f3 = _mm_add_ps(_f3, _mm_set1_ps(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                    __m128 _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_fmadd_ps(_c3, _beta, _f3);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
                        _f2 = _mm_add_ps(_f2, _mm_mul_ps(_c2, _beta));
                        _f3 = _mm_add_ps(_f3, _mm_mul_ps(_c3, _beta));
#endif
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }
            _mm_storeu_ps(p0, _f0);
            _mm_storeu_ps(p0 + out_hstep, _f1);
            _mm_storeu_ps(p0 + out_hstep * 2, _f2);
            _mm_storeu_ps(p0 + out_hstep * 3, _f3);
            p0 += 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _t0 = _mm_loadu_ps(pp);
            __m128 _t1 = _mm_loadu_ps(pp + 4);
            pp += 8;
            __m128 _tmp0 = _mm_shuffle_ps(_t0, _t0, _MM_SHUFFLE(3, 1, 2, 0));
            __m128 _tmp1 = _mm_shuffle_ps(_t1, _t1, _MM_SHUFFLE(0, 2, 3, 1));
            __m128 _c0v = _mm_unpacklo_ps(_tmp0, _tmp1);
            __m128 _c1v = _mm_unpackhi_ps(_tmp0, _tmp1);
            _c1v = _mm_shuffle_ps(_c1v, _c1v, _MM_SHUFFLE(2, 1, 0, 3));
            __m128 _f01 = _mm_unpacklo_ps(_c0v, _c1v);
            __m128 _f23 = _mm_unpackhi_ps(_c0v, _c1v);
            __m128 _f0 = _f01;
            __m128 _f1 = _mm_movehl_ps(_f01, _f01);
            __m128 _f2 = _f23;
            __m128 _f3 = _mm_movehl_ps(_f23, _f23);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                    _f2 = _mm_add_ps(_f2, _mm_set1_ps(c2));
                    _f3 = _mm_add_ps(_f3, _mm_set1_ps(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    __m128 _c1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep));
                    __m128 _c2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 2));
                    __m128 _c3 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 3));
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_fmadd_ps(_c3, _beta, _f3);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
                        _f2 = _mm_add_ps(_f2, _mm_mul_ps(_c2, _beta));
                        _f3 = _mm_add_ps(_f3, _mm_mul_ps(_c3, _beta));
#endif
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }
            _mm_storel_pi((__m64*)(p0), _f0);
            _mm_storel_pi((__m64*)(p0 + out_hstep), _f1);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 2), _f2);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 3), _f3);
            p0 += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m128 _f0 = _mm_loadu_ps(pp);
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_setr_ps(c0, c1, c2, c3));
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c;
                    if (broadcast_type_C == 3)
                        _c = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                    if (broadcast_type_C == 4)
                        _c = _mm_set1_ps(pC[0]);
                    if (beta == 1.f)
                        _f0 = _mm_add_ps(_f0, _c);
                    else
                        _f0 = _mm_comp_fmadd_ps(_c, _mm_set1_ps(beta), _f0);
                    pC++;
                }
            }
            if (alpha != 1.f)
                _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));
#if __AVX512F__
            __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
            _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32((int)out_hstep));
            _mm_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
#else
#ifdef _MSC_VER
            __declspec(align(16))
#else
            __attribute__((aligned(16)))
#endif
            float sum0[4];
            _mm_store_ps(sum0, _f0);
            p0[0] = sum0[0];
            p0[out_hstep] = sum0[1];
            p0[out_hstep * 2] = sum0[2];
            p0[out_hstep * 3] = sum0[3];
#endif
            p0++;
        }
    }

#endif // __SSE2__

    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;

        float c0 = 0.f;
        float c1 = c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                c1 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                c1 = pC[i + ii + 1];
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if (broadcast_type_C == 0 && beta != 1.f)
                c0 *= beta;
            if ((broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
            {
                c0 *= beta;
                c1 *= beta;
            }
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = _mm_loadu_ps(pp);
            __m128 _f1 = _mm_loadu_ps(pp + 4);
            __m128 _f2 = _mm_loadu_ps(pp + 8);
            __m128 _f3 = _mm_loadu_ps(pp + 12);
            pp += 16;
            _f2 = _mm_shuffle_ps(_f2, _f2, _MM_SHUFFLE(2, 3, 0, 1));
            _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f2);
            __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f2);
            __m128 _tmp2 = _mm_unpacklo_ps(_f1, _f3);
            __m128 _tmp3 = _mm_unpackhi_ps(_f1, _f3);
            _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
            _f1 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp2), _mm_castps_pd(_tmp3)));
            _f2 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
            _f3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp2), _mm_castps_pd(_tmp3)));
            _f2 = _mm_shuffle_ps(_f2, _f2, _MM_SHUFFLE(2, 3, 0, 1));
            _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 3, 0, 1));
            __m256 _f0x2 = combine4x2_ps(_f0, _f1);
            __m256 _f1x2 = combine4x2_ps(_f2, _f3);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = _mm256_set1_ps(c0);
                    _f0x2 = _mm256_add_ps(_f0x2, _c);
                    _f1x2 = _mm256_add_ps(_f1x2, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0x2 = _mm256_add_ps(_f0x2, _mm256_set1_ps(c0));
                    _f1x2 = _mm256_add_ps(_f1x2, _mm256_set1_ps(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = _mm256_loadu_ps(pC);
                    __m256 _c1 = _mm256_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0x2 = _mm256_add_ps(_f0x2, _c0);
                        _f1x2 = _mm256_add_ps(_f1x2, _c1);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
#if __FMA__
                        _f0x2 = _mm256_fmadd_ps(_c0, _beta, _f0x2);
                        _f1x2 = _mm256_fmadd_ps(_c1, _beta, _f1x2);
#else
                        _f0x2 = _mm256_add_ps(_f0x2, _mm256_mul_ps(_c0, _beta));
                        _f1x2 = _mm256_add_ps(_f1x2, _mm256_mul_ps(_c1, _beta));
#endif
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = _mm256_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm256_mul_ps(_c, _mm256_set1_ps(beta));
                    _f0x2 = _mm256_add_ps(_f0x2, _c);
                    _f1x2 = _mm256_add_ps(_f1x2, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0x2 = _mm256_mul_ps(_f0x2, _alpha);
                _f1x2 = _mm256_mul_ps(_f1x2, _alpha);
            }
            _mm256_storeu_ps(p0, _f0x2);
            _mm256_storeu_ps(p0 + out_hstep, _f1x2);
            p0 += 8;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_loadu_ps(pp + 0);
            __m128 _f1 = _mm_loadu_ps(pp + 4);
            pp += 8;
            __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f1);
            __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f1);
            _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
            _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp1), _mm_castps_pd(_tmp0)));
            _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(0, 3, 2, 1));
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
#endif
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }
            _mm_storeu_ps(p0, _f0);
            _mm_storeu_ps(p0 + out_hstep, _f1);
            p0 += 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pp + 0));
            __m128 _f1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pp + 2));
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    __m128 _c1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep));
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
#endif
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }
            _mm_storel_pi((__m64*)(p0), _f0);
            _mm_storel_pi((__m64*)(p0 + out_hstep), _f1);
            p0 += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m128 _f0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pp);
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_setr_ps(c0, c1, 0.f, 0.f));
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c;
                    if (broadcast_type_C == 3)
                        _c = _mm_setr_ps(pC[0], pC[c_hstep], 0.f, 0.f);
                    if (broadcast_type_C == 4)
                        _c = _mm_set1_ps(pC[0]);
                    if (beta == 1.f)
                        _f0 = _mm_add_ps(_f0, _c);
                    else
                        _f0 = _mm_comp_fmadd_ps(_c, _mm_set1_ps(beta), _f0);
                    pC++;
                }
            }
            if (alpha != 1.f)
                _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));
            _mm_store_ss(p0, _f0);
            _mm_store_ss(p0 + out_hstep, _mm_shuffle_ps(_f0, _f0, _MM_SHUFFLE(1, 1, 1, 1)));
            p0++;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0_0 = pp[0];
            float f0_1 = pp[1];
            float f1_0 = pp[2];
            float f1_1 = pp[3];
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0_0 += c0;
                    f0_1 += c0;
                    f1_0 += c0;
                    f1_1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0_0 += c0;
                    f0_1 += c0;
                    f1_0 += c1;
                    f1_1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    f0_0 += pC[0] * beta;
                    f0_1 += pC[1] * beta;
                    f1_0 += pC[c_hstep] * beta;
                    f1_1 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    f0_0 += pC[0] * beta;
                    f0_1 += pC[1] * beta;
                    f1_0 += pC[0] * beta;
                    f1_1 += pC[1] * beta;
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                f0_0 *= alpha;
                f0_1 *= alpha;
                f1_0 *= alpha;
                f1_1 *= alpha;
            }
            p0[0] = f0_0;
            p0[1] = f0_1;
            p0[out_hstep] = f1_0;
            p0[out_hstep + 1] = f1_1;

            p0 += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float f0_0 = pp[0];
            float f1_0 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0_0 += c0;
                    f1_0 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0_0 += c0;
                    f1_0 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    f0_0 += pC[0] * beta;
                    f1_0 += pC[c_hstep] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    f0_0 += pC[0] * beta;
                    f1_0 += pC[0] * beta;
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                f0_0 *= alpha;
                f1_0 *= alpha;
            }
            p0[0] = f0_0;
            p0[out_hstep] = f1_0;

            p0++;
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;

        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = pC[0];
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[i + ii];
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if ((broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
                c0 *= beta;
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = _mm256_loadu_ps(pp + 0);
            pp += 8;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = _mm256_set1_ps(c0);
                    _f0 = _mm256_add_ps(_f0, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(c0));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = _mm256_loadu_ps(pC);
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
#if __FMA__
                        _f0 = _mm256_fmadd_ps(_c0, _beta, _f0);
#else
                        _f0 = _mm256_add_ps(_f0, _mm256_mul_ps(_c0, _beta));
#endif
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = _mm256_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm256_mul_ps(_c, _mm256_set1_ps(beta));
                    _f0 = _mm256_add_ps(_f0, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
                _f0 = _mm256_mul_ps(_f0, _mm256_set1_ps(alpha));
            _mm256_storeu_ps(p0, _f0);
            p0 += 8;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_loadu_ps(pp + 0);
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadu_ps(pC);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
#endif
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
                _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));
            _mm_storeu_ps(p0, _f0);
            p0 += 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pp + 0));
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
#endif
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
                _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));
            _mm_storel_pi((__m64*)(p0), _f0);
            p0 += 2;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0_0 = pp[0];
            float f0_1 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0_0 += pC[0] * beta;
                    f0_1 += pC[1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0_0 += c0;
                    f0_1 += c0;
                }
            }
            if (alpha != 1.f)
            {
                f0_0 *= alpha;
                f0_1 *= alpha;
            }
            p0[0] = f0_0;
            p0[1] = f0_1;
            p0 += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float f0_0 = pp[0];
            pp += 1;
            if (pC)
            {
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0_0 += pC[0] * beta;
                    pC++;
                }
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    f0_0 += c0;
            }
            if (alpha != 1.f)
                f0_0 *= alpha;
            p0[0] = f0_0;
            p0++;
        }
    }
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_unpack_output_tile_wq_int8_avx512vnni(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        transpose_unpack_output_tile_wq_int8_avxvnniint8(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_unpack_output_tile_wq_int8_avxvnni(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_unpack_output_tile_wq_int8_avx2(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
        return;
    }
#endif

    const float* pC = C;
    const float* pp = topT;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;

    // topT microkernel lanes -> n0[m0..mMR-1], ..., nNR-1[m0..mMR-1]
    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* p0 = (float*)top_blob + j * out_hstep + i + ii;

        __m512 _c = _mm512_setzero_ps();
        __m512i _c_vindex = _mm512_setzero_si512();
        if (pC)
        {
            if (broadcast_type_C == 0)
                _c = _mm512_set1_ps(pC[0]);
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                _c = _mm512_loadu_ps(pC + i + ii);
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j;
                _c_vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _c_vindex = _mm512_mullo_epi32(_c_vindex, _mm512_set1_epi32((int)c_hstep));
            }
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if ((broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
                _c = _mm512_mul_ps(_c, _mm512_set1_ps(beta));
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m512 _f0 = _mm512_loadu_ps(pp + 0);
            __m512 _f1 = _mm512_loadu_ps(pp + 16);
            __m512 _f2 = _mm512_loadu_ps(pp + 32);
            __m512 _f3 = _mm512_loadu_ps(pp + 48);
            __m512 _f4 = _mm512_loadu_ps(pp + 64);
            __m512 _f5 = _mm512_loadu_ps(pp + 80);
            __m512 _f6 = _mm512_loadu_ps(pp + 96);
            __m512 _f7 = _mm512_loadu_ps(pp + 112);
            pp += 128;

            // from
            //      00 11 22 33  44 55 66 77  80 91 a2 b3  c4 d5 e6 f7
            //      01 12 23 30  45 56 67 74  81 92 a3 b0  c5 d6 e7 f4
            //      20 31 02 13  64 75 46 57  a0 b1 82 93  e4 f5 c6 d7
            //      21 32 03 10  65 76 47 54  a1 b2 83 90  e5 f6 c7 d4
            //      04 15 26 37  40 51 62 73  84 95 a6 b7  c0 d1 e2 f3
            //      05 16 27 34  41 52 63 70  85 96 a7 b4  c1 d2 e3 f0
            //      24 35 06 17  60 71 42 53  a4 b5 86 97  e0 f1 c2 d3
            //      25 36 07 14  61 72 43 50  a5 b6 87 94  e1 f2 c3 d0
            //
            // to
            //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
            //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
            //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
            //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
            //      04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
            //      05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
            //      06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
            //      07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
            _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
            __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
            __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
            __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
            __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
            __m512 _tmp4 = _mm512_unpacklo_ps(_f4, _f7);
            __m512 _tmp5 = _mm512_unpackhi_ps(_f4, _f7);
            __m512 _tmp6 = _mm512_unpacklo_ps(_f6, _f5);
            __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f5);
            _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
            _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
            _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
            _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
            _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
            _f5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
            _f6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
            _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
            _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
            _tmp0 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(0, 1, 1, 0));
            _tmp1 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
            _tmp2 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
            _tmp3 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
            _tmp4 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
            _tmp5 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
            _tmp6 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
            _tmp7 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));
            _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
            _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
            _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
            _f3 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
            _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 3, 1, 3));
            _f5 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
            _f6 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 3, 1, 3));
            _f7 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c);
                    _f1 = _mm512_add_ps(_f1, _c);
                    _f2 = _mm512_add_ps(_f2, _c);
                    _f3 = _mm512_add_ps(_f3, _c);
                    _f4 = _mm512_add_ps(_f4, _c);
                    _f5 = _mm512_add_ps(_f5, _c);
                    _f6 = _mm512_add_ps(_f6, _c);
                    _f7 = _mm512_add_ps(_f7, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c0 = _mm512_i32gather_ps(_c_vindex, pC, sizeof(float));
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_i32gather_ps(_c_vindex, pC + 1, sizeof(float));
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    __m512 _c2 = _mm512_i32gather_ps(_c_vindex, pC + 2, sizeof(float));
                    if (beta != 1.f)
                        _c2 = _mm512_mul_ps(_c2, _mm512_set1_ps(beta));
                    _f2 = _mm512_add_ps(_f2, _c2);
                    __m512 _c3 = _mm512_i32gather_ps(_c_vindex, pC + 3, sizeof(float));
                    if (beta != 1.f)
                        _c3 = _mm512_mul_ps(_c3, _mm512_set1_ps(beta));
                    _f3 = _mm512_add_ps(_f3, _c3);
                    __m512 _c4 = _mm512_i32gather_ps(_c_vindex, pC + 4, sizeof(float));
                    if (beta != 1.f)
                        _c4 = _mm512_mul_ps(_c4, _mm512_set1_ps(beta));
                    _f4 = _mm512_add_ps(_f4, _c4);
                    __m512 _c5 = _mm512_i32gather_ps(_c_vindex, pC + 5, sizeof(float));
                    if (beta != 1.f)
                        _c5 = _mm512_mul_ps(_c5, _mm512_set1_ps(beta));
                    _f5 = _mm512_add_ps(_f5, _c5);
                    __m512 _c6 = _mm512_i32gather_ps(_c_vindex, pC + 6, sizeof(float));
                    if (beta != 1.f)
                        _c6 = _mm512_mul_ps(_c6, _mm512_set1_ps(beta));
                    _f6 = _mm512_add_ps(_f6, _c6);
                    __m512 _c7 = _mm512_i32gather_ps(_c_vindex, pC + 7, sizeof(float));
                    if (beta != 1.f)
                        _c7 = _mm512_mul_ps(_c7, _mm512_set1_ps(beta));
                    _f7 = _mm512_add_ps(_f7, _c7);
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _c0 = _mm512_set1_ps(pC[0]);
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_set1_ps(pC[1]);
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    __m512 _c2 = _mm512_set1_ps(pC[2]);
                    if (beta != 1.f)
                        _c2 = _mm512_mul_ps(_c2, _mm512_set1_ps(beta));
                    _f2 = _mm512_add_ps(_f2, _c2);
                    __m512 _c3 = _mm512_set1_ps(pC[3]);
                    if (beta != 1.f)
                        _c3 = _mm512_mul_ps(_c3, _mm512_set1_ps(beta));
                    _f3 = _mm512_add_ps(_f3, _c3);
                    __m512 _c4 = _mm512_set1_ps(pC[4]);
                    if (beta != 1.f)
                        _c4 = _mm512_mul_ps(_c4, _mm512_set1_ps(beta));
                    _f4 = _mm512_add_ps(_f4, _c4);
                    __m512 _c5 = _mm512_set1_ps(pC[5]);
                    if (beta != 1.f)
                        _c5 = _mm512_mul_ps(_c5, _mm512_set1_ps(beta));
                    _f5 = _mm512_add_ps(_f5, _c5);
                    __m512 _c6 = _mm512_set1_ps(pC[6]);
                    if (beta != 1.f)
                        _c6 = _mm512_mul_ps(_c6, _mm512_set1_ps(beta));
                    _f6 = _mm512_add_ps(_f6, _c6);
                    __m512 _c7 = _mm512_set1_ps(pC[7]);
                    if (beta != 1.f)
                        _c7 = _mm512_mul_ps(_c7, _mm512_set1_ps(beta));
                    _f7 = _mm512_add_ps(_f7, _c7);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
                _f4 = _mm512_mul_ps(_f4, _alpha);
                _f5 = _mm512_mul_ps(_f5, _alpha);
                _f6 = _mm512_mul_ps(_f6, _alpha);
                _f7 = _mm512_mul_ps(_f7, _alpha);
            }
            _mm512_storeu_ps(p0, _f0);
            _mm512_storeu_ps(p0 + out_hstep, _f1);
            _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
            _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
            _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
            _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
            _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
            _mm512_storeu_ps(p0 + out_hstep * 7, _f7);
            p0 += out_hstep * 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512 _f0 = _mm512_loadu_ps(pp + 0);
            __m512 _f1 = _mm512_loadu_ps(pp + 16);
            __m512 _f2 = _mm512_loadu_ps(pp + 32);
            __m512 _f3 = _mm512_loadu_ps(pp + 48);
            pp += 64;
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
            __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
            __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
            __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
            _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
            _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
            _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
            _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c);
                    _f1 = _mm512_add_ps(_f1, _c);
                    _f2 = _mm512_add_ps(_f2, _c);
                    _f3 = _mm512_add_ps(_f3, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c0 = _mm512_i32gather_ps(_c_vindex, pC, sizeof(float));
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_i32gather_ps(_c_vindex, pC + 1, sizeof(float));
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    __m512 _c2 = _mm512_i32gather_ps(_c_vindex, pC + 2, sizeof(float));
                    if (beta != 1.f)
                        _c2 = _mm512_mul_ps(_c2, _mm512_set1_ps(beta));
                    _f2 = _mm512_add_ps(_f2, _c2);
                    __m512 _c3 = _mm512_i32gather_ps(_c_vindex, pC + 3, sizeof(float));
                    if (beta != 1.f)
                        _c3 = _mm512_mul_ps(_c3, _mm512_set1_ps(beta));
                    _f3 = _mm512_add_ps(_f3, _c3);
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _c0 = _mm512_set1_ps(pC[0]);
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_set1_ps(pC[1]);
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    __m512 _c2 = _mm512_set1_ps(pC[2]);
                    if (beta != 1.f)
                        _c2 = _mm512_mul_ps(_c2, _mm512_set1_ps(beta));
                    _f2 = _mm512_add_ps(_f2, _c2);
                    __m512 _c3 = _mm512_set1_ps(pC[3]);
                    if (beta != 1.f)
                        _c3 = _mm512_mul_ps(_c3, _mm512_set1_ps(beta));
                    _f3 = _mm512_add_ps(_f3, _c3);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
            }
            _mm512_storeu_ps(p0, _f0);
            _mm512_storeu_ps(p0 + out_hstep, _f1);
            _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
            _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
            p0 += out_hstep * 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512 _f0 = _mm512_loadu_ps(pp + 0);
            __m512 _f1 = _mm512_loadu_ps(pp + 16);
            pp += 32;
            __m512 _tmp0 = _mm512_permute_ps(_f0, _MM_SHUFFLE(3, 1, 2, 0));
            __m512 _tmp1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(0, 2, 3, 1));
            _f0 = _mm512_unpacklo_ps(_tmp0, _tmp1);
            _f1 = _mm512_unpackhi_ps(_tmp0, _tmp1);
            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c);
                    _f1 = _mm512_add_ps(_f1, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c0 = _mm512_i32gather_ps(_c_vindex, pC, sizeof(float));
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_i32gather_ps(_c_vindex, pC + 1, sizeof(float));
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _c0 = _mm512_set1_ps(pC[0]);
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    __m512 _c1 = _mm512_set1_ps(pC[1]);
                    if (beta != 1.f)
                        _c1 = _mm512_mul_ps(_c1, _mm512_set1_ps(beta));
                    _f1 = _mm512_add_ps(_f1, _c1);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
            }
            _mm512_storeu_ps(p0, _f0);
            _mm512_storeu_ps(p0 + out_hstep, _f1);
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj++)
        {
            __m512 _f0 = _mm512_loadu_ps(pp + 0);
            pp += 16;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c0 = _mm512_i32gather_ps(_c_vindex, pC, sizeof(float));
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _c0 = _mm512_set1_ps(pC[0]);
                    if (beta != 1.f)
                        _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0);
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
            }
            _mm512_storeu_ps(p0, _f0);
            p0 += out_hstep;
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    const float* pp1 = pp + max_jj * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = (float*)top_blob + j * out_hstep + i + ii;

        float c0 = 0.f;
        float c1 = c0;
        float c2 = c0;
        float c3 = c0;
        float c4 = c0;
        float c5 = c0;
        float c6 = c0;
        float c7 = c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                c1 = c0;
                c2 = c0;
                c3 = c0;
                c4 = c0;
                c5 = c0;
                c6 = c0;
                c7 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                c1 = pC[i + ii + 1];
                c2 = pC[i + ii + 2];
                c3 = pC[i + ii + 3];
                c4 = pC[i + ii + 4];
                c5 = pC[i + ii + 5];
                c6 = pC[i + ii + 6];
                c7 = pC[i + ii + 7];
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if (broadcast_type_C == 0 && beta != 1.f)
                c0 *= beta;
            if ((broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
            {
                c0 *= beta;
                c1 *= beta;
                c2 *= beta;
                c3 *= beta;
                c4 *= beta;
                c5 *= beta;
                c6 *= beta;
                c7 *= beta;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = _mm256_loadu_ps(pp + 0);
            __m256 _f1 = _mm256_loadu_ps(pp + 8);
            __m256 _f2 = _mm256_loadu_ps(pp + 16);
            __m256 _f3 = _mm256_loadu_ps(pp + 24);
            __m256 _f4 = _mm256_loadu_ps(pp + 32);
            __m256 _f5 = _mm256_loadu_ps(pp + 40);
            __m256 _f6 = _mm256_loadu_ps(pp + 48);
            __m256 _f7 = _mm256_loadu_ps(pp + 56);
            pp += 64;

            // from
            //      00 11 22 33 44 55 66 77
            //      01 12 23 30 45 56 67 74
            //      20 31 02 13 64 75 46 57
            //      21 32 03 10 65 76 47 54
            //      04 15 26 37 40 51 62 73
            //      05 16 27 34 41 52 63 70
            //      24 35 06 17 60 71 42 53
            //      25 36 07 14 61 72 43 50

            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            //      02 12 22 32 42 52 62 72
            //      03 13 23 33 43 53 63 73
            //      04 14 24 34 44 54 64 74
            //      05 15 25 35 45 55 65 75
            //      06 16 26 36 46 56 66 76
            //      07 17 27 37 47 57 67 77
            {
                __m256 _tmp0 = _f0;
                __m256 _tmp1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp2 = _f2;
                __m256 _tmp3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _f4;
                __m256 _tmp5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp6 = _f6;
                __m256 _tmp7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
                _f0 = _mm256_unpacklo_ps(_tmp0, _tmp3);
                _f1 = _mm256_unpackhi_ps(_tmp0, _tmp3);
                _f2 = _mm256_unpacklo_ps(_tmp2, _tmp1);
                _f3 = _mm256_unpackhi_ps(_tmp2, _tmp1);
                _f4 = _mm256_unpacklo_ps(_tmp4, _tmp7);
                _f5 = _mm256_unpackhi_ps(_tmp4, _tmp7);
                _f6 = _mm256_unpacklo_ps(_tmp6, _tmp5);
                _f7 = _mm256_unpackhi_ps(_tmp6, _tmp5);
                _tmp0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f0), _mm256_castps_pd(_f2)));
                _tmp1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f0), _mm256_castps_pd(_f2)));
                _tmp2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f3), _mm256_castps_pd(_f1)));
                _tmp3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f3), _mm256_castps_pd(_f1)));
                _tmp4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f4), _mm256_castps_pd(_f6)));
                _tmp5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f4), _mm256_castps_pd(_f6)));
                _tmp6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f7), _mm256_castps_pd(_f5)));
                _tmp7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f7), _mm256_castps_pd(_f5)));
                _tmp1 = _mm256_shuffle_ps(_tmp1, _tmp1, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp3 = _mm256_shuffle_ps(_tmp3, _tmp3, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp5 = _mm256_shuffle_ps(_tmp5, _tmp5, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp7 = _mm256_shuffle_ps(_tmp7, _tmp7, _MM_SHUFFLE(2, 1, 0, 3));
                _f0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 0));
                _f1 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 0));
                _f2 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 0));
                _f3 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 0));
                _f4 = _mm256_permute2f128_ps(_tmp4, _tmp0, _MM_SHUFFLE(0, 3, 0, 0));
                _f5 = _mm256_permute2f128_ps(_tmp5, _tmp1, _MM_SHUFFLE(0, 3, 0, 0));
                _f6 = _mm256_permute2f128_ps(_tmp6, _tmp2, _MM_SHUFFLE(0, 3, 0, 0));
                _f7 = _mm256_permute2f128_ps(_tmp7, _tmp3, _MM_SHUFFLE(0, 3, 0, 0));
            }
            transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = _mm256_set1_ps(c0);
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    _f2 = _mm256_add_ps(_f2, _c);
                    _f3 = _mm256_add_ps(_f3, _c);
                    _f4 = _mm256_add_ps(_f4, _c);
                    _f5 = _mm256_add_ps(_f5, _c);
                    _f6 = _mm256_add_ps(_f6, _c);
                    _f7 = _mm256_add_ps(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(c0));
                    _f1 = _mm256_add_ps(_f1, _mm256_set1_ps(c1));
                    _f2 = _mm256_add_ps(_f2, _mm256_set1_ps(c2));
                    _f3 = _mm256_add_ps(_f3, _mm256_set1_ps(c3));
                    _f4 = _mm256_add_ps(_f4, _mm256_set1_ps(c4));
                    _f5 = _mm256_add_ps(_f5, _mm256_set1_ps(c5));
                    _f6 = _mm256_add_ps(_f6, _mm256_set1_ps(c6));
                    _f7 = _mm256_add_ps(_f7, _mm256_set1_ps(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = _mm256_loadu_ps(pC);
                    __m256 _c1 = _mm256_loadu_ps(pC + c_hstep);
                    __m256 _c2 = _mm256_loadu_ps(pC + c_hstep * 2);
                    __m256 _c3 = _mm256_loadu_ps(pC + c_hstep * 3);
                    __m256 _c4 = _mm256_loadu_ps(pC + c_hstep * 4);
                    __m256 _c5 = _mm256_loadu_ps(pC + c_hstep * 5);
                    __m256 _c6 = _mm256_loadu_ps(pC + c_hstep * 6);
                    __m256 _c7 = _mm256_loadu_ps(pC + c_hstep * 7);
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                        _f2 = _mm256_add_ps(_f2, _c2);
                        _f3 = _mm256_add_ps(_f3, _c3);
                        _f4 = _mm256_add_ps(_f4, _c4);
                        _f5 = _mm256_add_ps(_f5, _c5);
                        _f6 = _mm256_add_ps(_f6, _c6);
                        _f7 = _mm256_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
#if __FMA__
                        _f0 = _mm256_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm256_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm256_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm256_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm256_fmadd_ps(_c7, _beta, _f7);
#else
                        _f0 = _mm256_add_ps(_f0, _mm256_mul_ps(_c0, _beta));
                        _f1 = _mm256_add_ps(_f1, _mm256_mul_ps(_c1, _beta));
                        _f2 = _mm256_add_ps(_f2, _mm256_mul_ps(_c2, _beta));
                        _f3 = _mm256_add_ps(_f3, _mm256_mul_ps(_c3, _beta));
                        _f4 = _mm256_add_ps(_f4, _mm256_mul_ps(_c4, _beta));
                        _f5 = _mm256_add_ps(_f5, _mm256_mul_ps(_c5, _beta));
                        _f6 = _mm256_add_ps(_f6, _mm256_mul_ps(_c6, _beta));
                        _f7 = _mm256_add_ps(_f7, _mm256_mul_ps(_c7, _beta));
#endif
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = _mm256_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm256_mul_ps(_c, _mm256_set1_ps(beta));
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    _f2 = _mm256_add_ps(_f2, _c);
                    _f3 = _mm256_add_ps(_f3, _c);
                    _f4 = _mm256_add_ps(_f4, _c);
                    _f5 = _mm256_add_ps(_f5, _c);
                    _f6 = _mm256_add_ps(_f6, _c);
                    _f7 = _mm256_add_ps(_f7, _c);
                    pC += 8;
                }
            }
            transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
                _f4 = _mm256_mul_ps(_f4, _alpha);
                _f5 = _mm256_mul_ps(_f5, _alpha);
                _f6 = _mm256_mul_ps(_f6, _alpha);
                _f7 = _mm256_mul_ps(_f7, _alpha);
            }
            _mm256_storeu_ps(p0, _f0);
            _mm256_storeu_ps(p0 + out_hstep, _f1);
            _mm256_storeu_ps(p0 + out_hstep * 2, _f2);
            _mm256_storeu_ps(p0 + out_hstep * 3, _f3);
            _mm256_storeu_ps(p0 + out_hstep * 4, _f4);
            _mm256_storeu_ps(p0 + out_hstep * 5, _f5);
            _mm256_storeu_ps(p0 + out_hstep * 6, _f6);
            _mm256_storeu_ps(p0 + out_hstep * 7, _f7);
            p0 += out_hstep * 8;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
#if __AVX2__
            __m256 _t0 = _mm256_loadu_ps(pp + 0);
            __m256 _t1 = _mm256_loadu_ps(pp + 8);
            __m256 _t2 = _mm256_loadu_ps(pp + 16);
            __m256 _t3 = _mm256_loadu_ps(pp + 24);
#else
            __m256 _t0 = combine4x2_ps(_mm_loadu_ps(pp + 0), _mm_loadu_ps(pp1 + 0));
            __m256 _t1 = combine4x2_ps(_mm_loadu_ps(pp + 4), _mm_loadu_ps(pp1 + 4));
            __m256 _t2 = combine4x2_ps(_mm_loadu_ps(pp + 8), _mm_loadu_ps(pp1 + 8));
            __m256 _t3 = combine4x2_ps(_mm_loadu_ps(pp + 12), _mm_loadu_ps(pp1 + 12));
#endif
#if __AVX2__
            pp += 32;
#else
            pp += 16;
            pp1 += 16;
#endif
            _t1 = _mm256_shuffle_ps(_t1, _t1, _MM_SHUFFLE(2, 1, 0, 3));
            _t3 = _mm256_shuffle_ps(_t3, _t3, _MM_SHUFFLE(2, 1, 0, 3));
            __m256 _tmp0 = _mm256_unpacklo_ps(_t0, _t3);
            __m256 _tmp1 = _mm256_unpackhi_ps(_t0, _t3);
            __m256 _tmp2 = _mm256_unpacklo_ps(_t2, _t1);
            __m256 _tmp3 = _mm256_unpackhi_ps(_t2, _t1);
            _t0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
            _t1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
            _t2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
            _t3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
            _t1 = _mm256_shuffle_ps(_t1, _t1, _MM_SHUFFLE(2, 1, 0, 3));
            _t3 = _mm256_shuffle_ps(_t3, _t3, _MM_SHUFFLE(2, 1, 0, 3));
            transpose8x4_ps(_t0, _t1, _t2, _t3);
            __m128 _f0 = _mm256_extractf128_ps(_t0, 0);
            __m128 _f1 = _mm256_extractf128_ps(_t0, 1);
            __m128 _f2 = _mm256_extractf128_ps(_t1, 0);
            __m128 _f3 = _mm256_extractf128_ps(_t1, 1);
            __m128 _f4 = _mm256_extractf128_ps(_t2, 0);
            __m128 _f5 = _mm256_extractf128_ps(_t2, 1);
            __m128 _f6 = _mm256_extractf128_ps(_t3, 0);
            __m128 _f7 = _mm256_extractf128_ps(_t3, 1);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    _f4 = _mm_add_ps(_f4, _c);
                    _f5 = _mm_add_ps(_f5, _c);
                    _f6 = _mm_add_ps(_f6, _c);
                    _f7 = _mm_add_ps(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                    _f2 = _mm_add_ps(_f2, _mm_set1_ps(c2));
                    _f3 = _mm_add_ps(_f3, _mm_set1_ps(c3));
                    _f4 = _mm_add_ps(_f4, _mm_set1_ps(c4));
                    _f5 = _mm_add_ps(_f5, _mm_set1_ps(c5));
                    _f6 = _mm_add_ps(_f6, _mm_set1_ps(c6));
                    _f7 = _mm_add_ps(_f7, _mm_set1_ps(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                    __m128 _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                    __m128 _c4 = _mm_loadu_ps(pC + c_hstep * 4);
                    __m128 _c5 = _mm_loadu_ps(pC + c_hstep * 5);
                    __m128 _c6 = _mm_loadu_ps(pC + c_hstep * 6);
                    __m128 _c7 = _mm_loadu_ps(pC + c_hstep * 7);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                        _f4 = _mm_add_ps(_f4, _c4);
                        _f5 = _mm_add_ps(_f5, _c5);
                        _f6 = _mm_add_ps(_f6, _c6);
                        _f7 = _mm_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm_fmadd_ps(_c7, _beta, _f7);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
                        _f2 = _mm_add_ps(_f2, _mm_mul_ps(_c2, _beta));
                        _f3 = _mm_add_ps(_f3, _mm_mul_ps(_c3, _beta));
                        _f4 = _mm_add_ps(_f4, _mm_mul_ps(_c4, _beta));
                        _f5 = _mm_add_ps(_f5, _mm_mul_ps(_c5, _beta));
                        _f6 = _mm_add_ps(_f6, _mm_mul_ps(_c6, _beta));
                        _f7 = _mm_add_ps(_f7, _mm_mul_ps(_c7, _beta));
#endif
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    _f4 = _mm_add_ps(_f4, _c);
                    _f5 = _mm_add_ps(_f5, _c);
                    _f6 = _mm_add_ps(_f6, _c);
                    _f7 = _mm_add_ps(_f7, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
                _f4 = _mm_mul_ps(_f4, _alpha);
                _f5 = _mm_mul_ps(_f5, _alpha);
                _f6 = _mm_mul_ps(_f6, _alpha);
                _f7 = _mm_mul_ps(_f7, _alpha);
            }
            {
                __m128 _r0 = _f0;
                __m128 _r1 = _f1;
                __m128 _r2 = _f2;
                __m128 _r3 = _f3;
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                __m128 _s0 = _f4;
                __m128 _s1 = _f5;
                __m128 _s2 = _f6;
                __m128 _s3 = _f7;
                _MM_TRANSPOSE4_PS(_s0, _s1, _s2, _s3);
                _mm256_storeu_ps(p0, _mm256_insertf128_ps(_mm256_castps128_ps256(_r0), _s0, 1));
                _mm256_storeu_ps(p0 + out_hstep, _mm256_insertf128_ps(_mm256_castps128_ps256(_r1), _s1, 1));
                _mm256_storeu_ps(p0 + out_hstep * 2, _mm256_insertf128_ps(_mm256_castps128_ps256(_r2), _s2, 1));
                _mm256_storeu_ps(p0 + out_hstep * 3, _mm256_insertf128_ps(_mm256_castps128_ps256(_r3), _s3, 1));
            }
            p0 += out_hstep * 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __AVX2__
            __m256 _t0 = _mm256_loadu_ps(pp);
            __m256 _t1 = _mm256_loadu_ps(pp + 8);
#else
            __m256 _t0 = combine4x2_ps(_mm_loadu_ps(pp), _mm_loadu_ps(pp1));
            __m256 _t1 = combine4x2_ps(_mm_loadu_ps(pp + 4), _mm_loadu_ps(pp1 + 4));
#endif
#if __AVX2__
            pp += 16;
#else
            pp += 8;
            pp1 += 8;
#endif
            __m256 _tmp0 = _mm256_shuffle_ps(_t0, _t0, _MM_SHUFFLE(3, 1, 2, 0));
            __m256 _tmp1 = _mm256_shuffle_ps(_t1, _t1, _MM_SHUFFLE(0, 2, 3, 1));
            _t0 = _mm256_unpacklo_ps(_tmp0, _tmp1);
            _t1 = _mm256_unpackhi_ps(_tmp0, _tmp1);
            _t1 = _mm256_shuffle_ps(_t1, _t1, _MM_SHUFFLE(2, 1, 0, 3));
            transpose8x2_ps(_t0, _t1);
            __m128 _r01 = _mm256_extractf128_ps(_t0, 0);
            __m128 _r23 = _mm256_extractf128_ps(_t0, 1);
            __m128 _r45 = _mm256_extractf128_ps(_t1, 0);
            __m128 _r67 = _mm256_extractf128_ps(_t1, 1);
            __m128 _f0 = _r01;
            __m128 _f1 = _mm_movehl_ps(_r01, _r01);
            __m128 _f2 = _r23;
            __m128 _f3 = _mm_movehl_ps(_r23, _r23);
            __m128 _f4 = _r45;
            __m128 _f5 = _mm_movehl_ps(_r45, _r45);
            __m128 _f6 = _r67;
            __m128 _f7 = _mm_movehl_ps(_r67, _r67);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    _f4 = _mm_add_ps(_f4, _c);
                    _f5 = _mm_add_ps(_f5, _c);
                    _f6 = _mm_add_ps(_f6, _c);
                    _f7 = _mm_add_ps(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                    _f2 = _mm_add_ps(_f2, _mm_set1_ps(c2));
                    _f3 = _mm_add_ps(_f3, _mm_set1_ps(c3));
                    _f4 = _mm_add_ps(_f4, _mm_set1_ps(c4));
                    _f5 = _mm_add_ps(_f5, _mm_set1_ps(c5));
                    _f6 = _mm_add_ps(_f6, _mm_set1_ps(c6));
                    _f7 = _mm_add_ps(_f7, _mm_set1_ps(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    __m128 _c1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep));
                    __m128 _c2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 2));
                    __m128 _c3 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 3));
                    __m128 _c4 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 4));
                    __m128 _c5 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 5));
                    __m128 _c6 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 6));
                    __m128 _c7 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 7));
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                        _f4 = _mm_add_ps(_f4, _c4);
                        _f5 = _mm_add_ps(_f5, _c5);
                        _f6 = _mm_add_ps(_f6, _c6);
                        _f7 = _mm_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm_fmadd_ps(_c7, _beta, _f7);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
                        _f2 = _mm_add_ps(_f2, _mm_mul_ps(_c2, _beta));
                        _f3 = _mm_add_ps(_f3, _mm_mul_ps(_c3, _beta));
                        _f4 = _mm_add_ps(_f4, _mm_mul_ps(_c4, _beta));
                        _f5 = _mm_add_ps(_f5, _mm_mul_ps(_c5, _beta));
                        _f6 = _mm_add_ps(_f6, _mm_mul_ps(_c6, _beta));
                        _f7 = _mm_add_ps(_f7, _mm_mul_ps(_c7, _beta));
#endif
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    _f4 = _mm_add_ps(_f4, _c);
                    _f5 = _mm_add_ps(_f5, _c);
                    _f6 = _mm_add_ps(_f6, _c);
                    _f7 = _mm_add_ps(_f7, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
                _f4 = _mm_mul_ps(_f4, _alpha);
                _f5 = _mm_mul_ps(_f5, _alpha);
                _f6 = _mm_mul_ps(_f6, _alpha);
                _f7 = _mm_mul_ps(_f7, _alpha);
            }
            {
                __m128 _r0 = _f0;
                __m128 _r1 = _f1;
                __m128 _r2 = _f2;
                __m128 _r3 = _f3;
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                __m128 _s0 = _f4;
                __m128 _s1 = _f5;
                __m128 _s2 = _f6;
                __m128 _s3 = _f7;
                _MM_TRANSPOSE4_PS(_s0, _s1, _s2, _s3);
                _mm256_storeu_ps(p0, _mm256_insertf128_ps(_mm256_castps128_ps256(_r0), _s0, 1));
                _mm256_storeu_ps(p0 + out_hstep, _mm256_insertf128_ps(_mm256_castps128_ps256(_r1), _s1, 1));
            }
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj++)
        {
#if __AVX2__
            __m128 _f03 = _mm_loadu_ps(pp);
            __m128 _f47 = _mm_loadu_ps(pp + 4);
#else
            __m128 _f03 = _mm_loadu_ps(pp);
            __m128 _f47 = _mm_loadu_ps(pp1);
#endif
#if __AVX2__
            pp += 8;
#else
            pp += 4;
            pp1 += 4;
#endif
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f03 = _mm_add_ps(_f03, _c);
                    _f47 = _mm_add_ps(_f47, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c03 = _mm_setr_ps(c0, c1, c2, c3);
                    __m128 _c47 = _mm_setr_ps(c4, c5, c6, c7);
                    _f03 = _mm_add_ps(_f03, _c03);
                    _f47 = _mm_add_ps(_f47, _c47);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c03;
                    __m128 _c47;
                    if (broadcast_type_C == 3)
                    {
                        _c03 = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                        _c47 = _mm_setr_ps(pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]);
                    }
                    if (broadcast_type_C == 4)
                    {
                        _c03 = _mm_set1_ps(pC[0]);
                        _c47 = _c03;
                    }
                    if (beta == 1.f)
                    {
                        _f03 = _mm_add_ps(_f03, _c03);
                        _f47 = _mm_add_ps(_f47, _c47);
                    }
                    else
                    {
#if __FMA__
                        __m128 _beta = _mm_set1_ps(beta);
                        _f03 = _mm_fmadd_ps(_c03, _beta, _f03);
                        _f47 = _mm_fmadd_ps(_c47, _beta, _f47);
#else
                        _f03 = _mm_add_ps(_f03, _mm_mul_ps(_c03, _mm_set1_ps(beta)));
                        _f47 = _mm_add_ps(_f47, _mm_mul_ps(_c47, _mm_set1_ps(beta)));
#endif
                    }
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f03 = _mm_mul_ps(_f03, _alpha);
                _f47 = _mm_mul_ps(_f47, _alpha);
            }
            _mm256_storeu_ps(p0, _mm256_insertf128_ps(_mm256_castps128_ps256(_f03), _f47, 1));
            p0 += out_hstep;
        }
#if !__AVX2__
        pp = pp1;
        pp1 = pp + max_jj * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = (float*)top_blob + j * out_hstep + i + ii;

        float c0 = 0.f;
        float c1 = c0;
        float c2 = c0;
        float c3 = c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                c1 = c0;
                c2 = c0;
                c3 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                c1 = pC[i + ii + 1];
                c2 = pC[i + ii + 2];
                c3 = pC[i + ii + 3];
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if (broadcast_type_C == 0 && beta != 1.f)
                c0 *= beta;
            if ((broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
            {
                c0 *= beta;
                c1 *= beta;
                c2 *= beta;
                c3 *= beta;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = _mm256_loadu_ps(pp + 0);
            __m256 _f1 = _mm256_loadu_ps(pp + 8);
            __m256 _f2 = _mm256_loadu_ps(pp + 16);
            __m256 _f3 = _mm256_loadu_ps(pp + 24);
            pp += 32;
            __m128 _f00 = _mm256_castps256_ps128(_f0);
            __m128 _f01 = _mm256_castps256_ps128(_f1);
            __m128 _f02 = _mm256_castps256_ps128(_f2);
            __m128 _f03 = _mm256_castps256_ps128(_f3);
            __m128 _f10 = _mm256_extractf128_ps(_f0, 1);
            __m128 _f11 = _mm256_extractf128_ps(_f1, 1);
            __m128 _f12 = _mm256_extractf128_ps(_f2, 1);
            __m128 _f13 = _mm256_extractf128_ps(_f3, 1);
            {
                _f01 = _mm_shuffle_ps(_f01, _f01, _MM_SHUFFLE(2, 1, 0, 3));
                _f03 = _mm_shuffle_ps(_f03, _f03, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f00, _f03);
                __m128 _tmp1 = _mm_unpackhi_ps(_f00, _f03);
                __m128 _tmp2 = _mm_unpacklo_ps(_f02, _f01);
                __m128 _tmp3 = _mm_unpackhi_ps(_f02, _f01);
                _f00 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f01 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f02 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f03 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f01 = _mm_shuffle_ps(_f01, _f01, _MM_SHUFFLE(2, 1, 0, 3));
                _f03 = _mm_shuffle_ps(_f03, _f03, _MM_SHUFFLE(2, 1, 0, 3));
                _MM_TRANSPOSE4_PS(_f00, _f01, _f02, _f03);
            }
            {
                _f11 = _mm_shuffle_ps(_f11, _f11, _MM_SHUFFLE(2, 1, 0, 3));
                _f13 = _mm_shuffle_ps(_f13, _f13, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f10, _f13);
                __m128 _tmp1 = _mm_unpackhi_ps(_f10, _f13);
                __m128 _tmp2 = _mm_unpacklo_ps(_f12, _f11);
                __m128 _tmp3 = _mm_unpackhi_ps(_f12, _f11);
                _f10 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f11 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f11 = _mm_shuffle_ps(_f11, _f11, _MM_SHUFFLE(2, 1, 0, 3));
                _f13 = _mm_shuffle_ps(_f13, _f13, _MM_SHUFFLE(2, 1, 0, 3));
                _MM_TRANSPOSE4_PS(_f10, _f11, _f12, _f13);
            }
            _f0 = combine4x2_ps(_f00, _f10);
            _f1 = combine4x2_ps(_f01, _f11);
            _f2 = combine4x2_ps(_f02, _f12);
            _f3 = combine4x2_ps(_f03, _f13);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = _mm256_set1_ps(c0);
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    _f2 = _mm256_add_ps(_f2, _c);
                    _f3 = _mm256_add_ps(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(c0));
                    _f1 = _mm256_add_ps(_f1, _mm256_set1_ps(c1));
                    _f2 = _mm256_add_ps(_f2, _mm256_set1_ps(c2));
                    _f3 = _mm256_add_ps(_f3, _mm256_set1_ps(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = _mm256_loadu_ps(pC);
                    __m256 _c1 = _mm256_loadu_ps(pC + c_hstep);
                    __m256 _c2 = _mm256_loadu_ps(pC + c_hstep * 2);
                    __m256 _c3 = _mm256_loadu_ps(pC + c_hstep * 3);
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                        _f2 = _mm256_add_ps(_f2, _c2);
                        _f3 = _mm256_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
#if __FMA__
                        _f0 = _mm256_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_fmadd_ps(_c3, _beta, _f3);
#else
                        _f0 = _mm256_add_ps(_f0, _mm256_mul_ps(_c0, _beta));
                        _f1 = _mm256_add_ps(_f1, _mm256_mul_ps(_c1, _beta));
                        _f2 = _mm256_add_ps(_f2, _mm256_mul_ps(_c2, _beta));
                        _f3 = _mm256_add_ps(_f3, _mm256_mul_ps(_c3, _beta));
#endif
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = _mm256_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm256_mul_ps(_c, _mm256_set1_ps(beta));
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    _f2 = _mm256_add_ps(_f2, _c);
                    _f3 = _mm256_add_ps(_f3, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
            }
            {
                __m128 _r0 = _mm256_castps256_ps128(_f0);
                __m128 _r1 = _mm256_castps256_ps128(_f1);
                __m128 _r2 = _mm256_castps256_ps128(_f2);
                __m128 _r3 = _mm256_castps256_ps128(_f3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_storeu_ps(p0, _r0);
                _mm_storeu_ps(p0 + out_hstep, _r1);
                _mm_storeu_ps(p0 + out_hstep * 2, _r2);
                _mm_storeu_ps(p0 + out_hstep * 3, _r3);
            }
            {
                __m128 _r0 = _mm256_extractf128_ps(_f0, 1);
                __m128 _r1 = _mm256_extractf128_ps(_f1, 1);
                __m128 _r2 = _mm256_extractf128_ps(_f2, 1);
                __m128 _r3 = _mm256_extractf128_ps(_f3, 1);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_storeu_ps(p0 + out_hstep * 4, _r0);
                _mm_storeu_ps(p0 + out_hstep * 5, _r1);
                _mm_storeu_ps(p0 + out_hstep * 6, _r2);
                _mm_storeu_ps(p0 + out_hstep * 7, _r3);
            }
            p0 += out_hstep * 8;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_loadu_ps(pp + 0);
            __m128 _f1 = _mm_loadu_ps(pp + 4);
            __m128 _f2 = _mm_loadu_ps(pp + 8);
            __m128 _f3 = _mm_loadu_ps(pp + 12);
            pp += 16;
            {
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f3);
                __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f3);
                __m128 _tmp2 = _mm_unpacklo_ps(_f2, _f1);
                __m128 _tmp3 = _mm_unpackhi_ps(_f2, _f1);
                _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
            }
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                    _f2 = _mm_add_ps(_f2, _mm_set1_ps(c2));
                    _f3 = _mm_add_ps(_f3, _mm_set1_ps(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                    __m128 _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_fmadd_ps(_c3, _beta, _f3);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
                        _f2 = _mm_add_ps(_f2, _mm_mul_ps(_c2, _beta));
                        _f3 = _mm_add_ps(_f3, _mm_mul_ps(_c3, _beta));
#endif
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }
            {
                __m128 _r0 = _f0;
                __m128 _r1 = _f1;
                __m128 _r2 = _f2;
                __m128 _r3 = _f3;
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_storeu_ps(p0, _r0);
                _mm_storeu_ps(p0 + out_hstep, _r1);
                _mm_storeu_ps(p0 + out_hstep * 2, _r2);
                _mm_storeu_ps(p0 + out_hstep * 3, _r3);
            }
            p0 += out_hstep * 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _t0 = _mm_loadu_ps(pp);
            __m128 _t1 = _mm_loadu_ps(pp + 4);
            pp += 8;
            __m128 _tmp0 = _mm_shuffle_ps(_t0, _t0, _MM_SHUFFLE(3, 1, 2, 0));
            __m128 _tmp1 = _mm_shuffle_ps(_t1, _t1, _MM_SHUFFLE(0, 2, 3, 1));
            __m128 _c0v = _mm_unpacklo_ps(_tmp0, _tmp1);
            __m128 _c1v = _mm_unpackhi_ps(_tmp0, _tmp1);
            _c1v = _mm_shuffle_ps(_c1v, _c1v, _MM_SHUFFLE(2, 1, 0, 3));
            __m128 _f01 = _mm_unpacklo_ps(_c0v, _c1v);
            __m128 _f23 = _mm_unpackhi_ps(_c0v, _c1v);
            __m128 _f0 = _f01;
            __m128 _f1 = _mm_movehl_ps(_f01, _f01);
            __m128 _f2 = _f23;
            __m128 _f3 = _mm_movehl_ps(_f23, _f23);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                    _f2 = _mm_add_ps(_f2, _mm_set1_ps(c2));
                    _f3 = _mm_add_ps(_f3, _mm_set1_ps(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    __m128 _c1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep));
                    __m128 _c2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 2));
                    __m128 _c3 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep * 3));
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_fmadd_ps(_c3, _beta, _f3);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
                        _f2 = _mm_add_ps(_f2, _mm_mul_ps(_c2, _beta));
                        _f3 = _mm_add_ps(_f3, _mm_mul_ps(_c3, _beta));
#endif
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    _f2 = _mm_add_ps(_f2, _c);
                    _f3 = _mm_add_ps(_f3, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }
            {
                __m128 _r0 = _f0;
                __m128 _r1 = _f1;
                __m128 _r2 = _f2;
                __m128 _r3 = _f3;
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_storeu_ps(p0, _r0);
                _mm_storeu_ps(p0 + out_hstep, _r1);
            }
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m128 _f = _mm_loadu_ps(pp);
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0)
                    _f = _mm_add_ps(_f, _mm_set1_ps(c0));
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = _mm_add_ps(_f, _mm_setr_ps(c0, c1, c2, c3));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c;
                    if (broadcast_type_C == 3)
                        _c = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                    if (broadcast_type_C == 4)
                        _c = _mm_set1_ps(pC[0]);
                    if (beta == 1.f)
                    {
                        _f = _mm_add_ps(_f, _c);
                    }
                    else
                    {
#if __FMA__
                        _f = _mm_fmadd_ps(_c, _mm_set1_ps(beta), _f);
#else
                        _f = _mm_add_ps(_f, _mm_mul_ps(_c, _mm_set1_ps(beta)));
#endif
                    }
                    pC++;
                }
            }
            if (alpha != 1.f)
                _f = _mm_mul_ps(_f, _mm_set1_ps(alpha));
            _mm_storeu_ps(p0, _f);
            p0 += out_hstep;
        }
    }

#endif // __SSE2__

    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = (float*)top_blob + j * out_hstep + i + ii;

        float c0 = 0.f;
        float c1 = c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                c1 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                c1 = pC[i + ii + 1];
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if (broadcast_type_C == 0 && beta != 1.f)
                c0 *= beta;
            if ((broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
            {
                c0 *= beta;
                c1 *= beta;
            }
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _t0 = _mm_loadu_ps(pp);
            __m128 _t1 = _mm_loadu_ps(pp + 4);
            __m128 _t2 = _mm_loadu_ps(pp + 8);
            __m128 _t3 = _mm_loadu_ps(pp + 12);
            pp += 16;
            _t2 = _mm_shuffle_ps(_t2, _t2, _MM_SHUFFLE(2, 3, 0, 1));
            _t3 = _mm_shuffle_ps(_t3, _t3, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 _tmp0x = _mm_unpacklo_ps(_t0, _t2);
            __m128 _tmp1x = _mm_unpackhi_ps(_t0, _t2);
            __m128 _tmp2x = _mm_unpacklo_ps(_t1, _t3);
            __m128 _tmp3x = _mm_unpackhi_ps(_t1, _t3);
            _t0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0x), _mm_castps_pd(_tmp1x)));
            _t1 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp2x), _mm_castps_pd(_tmp3x)));
            _t2 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0x), _mm_castps_pd(_tmp1x)));
            _t3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp2x), _mm_castps_pd(_tmp3x)));
            _t2 = _mm_shuffle_ps(_t2, _t2, _MM_SHUFFLE(2, 3, 0, 1));
            _t3 = _mm_shuffle_ps(_t3, _t3, _MM_SHUFFLE(2, 3, 0, 1));
            __m256 _f0 = combine4x2_ps(_t0, _t1);
            __m256 _f1 = combine4x2_ps(_t2, _t3);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = _mm256_set1_ps(c0);
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(c0));
                    _f1 = _mm256_add_ps(_f1, _mm256_set1_ps(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = _mm256_loadu_ps(pC);
                    __m256 _c1 = _mm256_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
#if __FMA__
                        _f0 = _mm256_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_fmadd_ps(_c1, _beta, _f1);
#else
                        _f0 = _mm256_add_ps(_f0, _mm256_mul_ps(_c0, _beta));
                        _f1 = _mm256_add_ps(_f1, _mm256_mul_ps(_c1, _beta));
#endif
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = _mm256_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm256_mul_ps(_c, _mm256_set1_ps(beta));
                    _f0 = _mm256_add_ps(_f0, _c);
                    _f1 = _mm256_add_ps(_f1, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
            }
            __m256 _tmp0 = _mm256_unpacklo_ps(_f0, _f1);
            __m256 _tmp1 = _mm256_unpackhi_ps(_f0, _f1);
            __m128 _r0 = _mm256_castps256_ps128(_tmp0);
            __m128 _r1 = _mm256_castps256_ps128(_tmp1);
            _mm_storel_pi((__m64*)p0, _r0);
            _mm_storeh_pi((__m64*)(p0 + out_hstep), _r0);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 2), _r1);
            _mm_storeh_pi((__m64*)(p0 + out_hstep * 3), _r1);
            _r0 = _mm256_extractf128_ps(_tmp0, 1);
            _r1 = _mm256_extractf128_ps(_tmp1, 1);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 4), _r0);
            _mm_storeh_pi((__m64*)(p0 + out_hstep * 5), _r0);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 6), _r1);
            _mm_storeh_pi((__m64*)(p0 + out_hstep * 7), _r1);
            p0 += out_hstep * 8;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _t0 = _mm_loadu_ps(pp + 0);
            __m128 _t1 = _mm_loadu_ps(pp + 4);
            pp += 8;
            __m128 _tmp0x = _mm_unpacklo_ps(_t0, _t1);
            __m128 _tmp1x = _mm_unpackhi_ps(_t0, _t1);
            __m128 _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0x), _mm_castps_pd(_tmp1x)));
            __m128 _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp1x), _mm_castps_pd(_tmp0x)));
            _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(0, 3, 2, 1));
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
#endif
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadu_ps(pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }
            __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f1);
            __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f1);
            _mm_storel_pi((__m64*)p0, _tmp0);
            _mm_storeh_pi((__m64*)(p0 + out_hstep), _tmp0);
            _mm_storel_pi((__m64*)(p0 + out_hstep * 2), _tmp1);
            _mm_storeh_pi((__m64*)(p0 + out_hstep * 3), _tmp1);
            p0 += out_hstep * 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pp + 0));
            __m128 _f1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pp + 2));
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = _mm_set1_ps(c0);
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _mm_set1_ps(c0));
                    _f1 = _mm_add_ps(_f1, _mm_set1_ps(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    __m128 _c1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pC + c_hstep));
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
#if __FMA__
                        _f0 = _mm_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_fmadd_ps(_c1, _beta, _f1);
#else
                        _f0 = _mm_add_ps(_f0, _mm_mul_ps(_c0, _beta));
                        _f1 = _mm_add_ps(_f1, _mm_mul_ps(_c1, _beta));
#endif
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta != 1.f)
                        _c = _mm_mul_ps(_c, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c);
                    _f1 = _mm_add_ps(_f1, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }
            __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f1);
            _mm_storel_pi((__m64*)p0, _tmp0);
            _mm_storeh_pi((__m64*)(p0 + out_hstep), _tmp0);
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m128 _f = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pp);
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0)
                    _f = _mm_add_ps(_f, _mm_set1_ps(c0));
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = _mm_add_ps(_f, _mm_setr_ps(c0, c1, 0.f, 0.f));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c;
                    if (broadcast_type_C == 3)
                        _c = _mm_setr_ps(pC[0], pC[c_hstep], 0.f, 0.f);
                    if (broadcast_type_C == 4)
                        _c = _mm_set1_ps(pC[0]);
                    if (beta == 1.f)
                    {
                        _f = _mm_add_ps(_f, _c);
                    }
                    else
                    {
#if __FMA__
                        _f = _mm_fmadd_ps(_c, _mm_set1_ps(beta), _f);
#else
                        _f = _mm_add_ps(_f, _mm_mul_ps(_c, _mm_set1_ps(beta)));
#endif
                    }
                    pC++;
                }
            }
            if (alpha != 1.f)
                _f = _mm_mul_ps(_f, _mm_set1_ps(alpha));
            _mm_storel_pi((__m64*)p0, _f);
            p0 += out_hstep;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0_0 = pp[0];
            float f0_1 = pp[1];
            float f1_0 = pp[2];
            float f1_1 = pp[3];
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0_0 += c0;
                    f0_1 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0_0 += pC[0] * beta;
                    f0_1 += pC[1] * beta;
                }
            }
            if (alpha != 1.f)
            {
                f0_0 *= alpha;
                f0_1 *= alpha;
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f1_0 += c0;
                    f1_1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f1_0 += c1;
                    f1_1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    f1_0 += pC[c_hstep] * beta;
                    f1_1 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    f1_0 += pC[0] * beta;
                    f1_1 += pC[1] * beta;
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                f1_0 *= alpha;
                f1_1 *= alpha;
            }

            p0[0] = f0_0;
            p0[1] = f1_0;
            p0[out_hstep] = f0_1;
            p0[out_hstep + 1] = f1_1;

            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float f0_0 = pp[0];
            float f1_0 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    f0_0 += c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    f0_0 += pC[0] * beta;
            }
            if (alpha != 1.f)
                f0_0 *= alpha;

            if (pC)
            {
                if (broadcast_type_C == 0)
                    f1_0 += c0;
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    f1_0 += c1;
                if (broadcast_type_C == 3)
                {
                    f1_0 += pC[c_hstep] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    f1_0 += pC[0] * beta;
                    pC++;
                }
            }
            if (alpha != 1.f)
                f1_0 *= alpha;

            p0[0] = f0_0;
            p0[1] = f1_0;

            p0 += out_hstep;
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        float* p0 = (float*)top_blob + j * out_hstep + i + ii;

        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = pC[0];
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[i + ii];
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
            if ((broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2) && beta != 1.f)
                c0 *= beta;
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f = _mm256_loadu_ps(pp);
            pp += 8;
            if (pC)
            {
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m256 _c = _mm256_loadu_ps(pC);
                    if (beta == 1.f)
                    {
                        _f = _mm256_add_ps(_f, _c);
                    }
                    else
                    {
#if __FMA__
                        _f = _mm256_fmadd_ps(_c, _mm256_set1_ps(beta), _f);
#else
                        _f = _mm256_add_ps(_f, _mm256_mul_ps(_c, _mm256_set1_ps(beta)));
#endif
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f = _mm256_add_ps(_f, _mm256_set1_ps(c0));
                }
            }
            if (alpha != 1.f)
                _f = _mm256_mul_ps(_f, _mm256_set1_ps(alpha));

            if (out_hstep == 1)
            {
                _mm256_storeu_ps(p0, _f);
            }
            else
            {
                __m128 _r0 = _mm256_castps256_ps128(_f);
                __m128 _r1 = _mm256_extractf128_ps(_f, 1);
                _mm_store_ss(p0, _r0);
                _r0 = _mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep, _r0);
                _r0 = _mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep * 2, _r0);
                _r0 = _mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep * 3, _r0);
                _mm_store_ss(p0 + out_hstep * 4, _r1);
                _r1 = _mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep * 5, _r1);
                _r1 = _mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep * 6, _r1);
                _r1 = _mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep * 7, _r1);
            }
            p0 += out_hstep * 8;
        }
#endif // __AVX512F__
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f = _mm_loadu_ps(pp);
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadu_ps(pC);
                    if (beta == 1.f)
                    {
                        _f = _mm_add_ps(_f, _c);
                    }
                    else
                    {
#if __FMA__
                        _f = _mm_fmadd_ps(_c, _mm_set1_ps(beta), _f);
#else
                        _f = _mm_add_ps(_f, _mm_mul_ps(_c, _mm_set1_ps(beta)));
#endif
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f = _mm_add_ps(_f, _mm_set1_ps(c0));
                }
            }
            if (alpha != 1.f)
                _f = _mm_mul_ps(_f, _mm_set1_ps(alpha));

            if (out_hstep == 1)
            {
                _mm_storeu_ps(p0, _f);
            }
            else
            {
                _mm_store_ss(p0, _f);
                _f = _mm_shuffle_ps(_f, _f, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep, _f);
                _f = _mm_shuffle_ps(_f, _f, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep * 2, _f);
                _f = _mm_shuffle_ps(_f, _f, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep * 3, _f);
            }
            p0 += out_hstep * 4;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pp);
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pC);
                    if (beta == 1.f)
                    {
                        _f = _mm_add_ps(_f, _c);
                    }
                    else
                    {
#if __FMA__
                        _f = _mm_fmadd_ps(_c, _mm_set1_ps(beta), _f);
#else
                        _f = _mm_add_ps(_f, _mm_mul_ps(_c, _mm_set1_ps(beta)));
#endif
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f = _mm_add_ps(_f, _mm_set1_ps(c0));
                }
            }
            if (alpha != 1.f)
                _f = _mm_mul_ps(_f, _mm_set1_ps(alpha));

            if (out_hstep == 1)
            {
                _mm_storel_pi((__m64*)p0, _f);
            }
            else
            {
                _mm_store_ss(p0, _f);
                _f = _mm_shuffle_ps(_f, _f, _MM_SHUFFLE(0, 3, 2, 1));
                _mm_store_ss(p0 + out_hstep, _f);
            }
            p0 += out_hstep * 2;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0_0 = pp[0];
            float f0_1 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0_0 += pC[0] * beta;
                    f0_1 += pC[1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0_0 += c0;
                    f0_1 += c0;
                }
            }
            if (alpha != 1.f)
            {
                f0_0 *= alpha;
                f0_1 *= alpha;
            }
            p0[0] = f0_0;
            p0[out_hstep] = f0_1;
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float f0_0 = pp[0];
            pp += 1;
            if (pC)
            {
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0_0 += beta == 1.f ? pC[0] : pC[0] * beta;
                    pC++;
                }
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    f0_0 += c0;
            }
            if (alpha != 1.f)
                f0_0 *= alpha;
            p0[0] = f0_0;
            p0 += out_hstep;
        }
    }
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(float)));
    TILE_K = std::max(block_size, tile_size / block_size * block_size);
    if (K > 0)
        TILE_K = std::min(K, TILE_K);

#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    TILE_M = std::max(16, tile_size / 16 * 16);
    TILE_N = std::max(8, tile_size / 8 * 8);
#else
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(4, tile_size / 4 * 4);
#endif // __AVX512F__
#else
#if __SSE2__
    TILE_M = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
#endif // __SSE2__
    TILE_N = std::max(2, tile_size / 2 * 2);
#endif

    if (K > 0 && (K + TILE_K - 1) / TILE_K == 1)
    {
        tile_size = std::max(1, (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K));

#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        TILE_M = std::max(16, tile_size / 16 * 16);
        TILE_N = std::max(8, tile_size / 8 * 8);
#else
        TILE_M = std::max(8, tile_size / 8 * 8);
        TILE_N = std::max(4, tile_size / 4 * 4);
#endif // __AVX512F__
#else
#if __SSE2__
        TILE_M = std::max(4, tile_size / 4 * 4);
#else
        TILE_M = std::max(2, tile_size / 2 * 2);
#endif // __SSE2__
        TILE_N = std::max(2, tile_size / 2 * 2);
#endif
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        const int nn_M = (M + TILE_M - 1) / TILE_M;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#endif // __AVX512F__
#else
#if __SSE2__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif // __SSE2__
#endif
    }

    if (N > 0)
    {
        const int nn_N = (N + TILE_N - 1) / TILE_N;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#endif // __AVX512F__
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2);
#endif
    }

    if (nT > 1)
    {
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 15) / 16 * 16);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#endif // __AVX512F__
#else
#if __SSE2__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif // __SSE2__
#endif
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        TILE_M = (constant_TILE_M + 15) / 16 * 16;
#else
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#endif // __AVX512F__
#else
#if __SSE2__
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif // __SSE2__
#endif
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
#endif
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, constant_TILE_K / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(K, TILE_K);
    }
}

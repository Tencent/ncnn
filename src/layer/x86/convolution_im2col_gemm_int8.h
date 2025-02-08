// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void convolution_im2col_input_tile_int8_avx512vnni(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void convolution_im2col_input_tile_int8_avxvnniint8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVX512VNNI__
void convolution_im2col_input_tile_int8_avxvnni(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void convolution_im2col_input_tile_int8_avx2(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h);
void unpack_output_tile_int32_avx2(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
#endif

// gemm_x86.h
#if NCNN_RUNTIME_CPU && __AVX512F__
namespace Gemm_x86_avx512_utility {
#elif NCNN_RUNTIME_CPU && __FMA__
namespace Gemm_x86_fma_utility {
#elif NCNN_RUNTIME_CPU && __AVX__
namespace Gemm_x86_avx_utility {
#else
namespace Gemm_x86_utility {
#endif
void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
}

static void convolution_im2col_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch

#if NCNN_RUNTIME_CPU && __AVX512F__
    Gemm_x86_avx512_utility::pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
#elif NCNN_RUNTIME_CPU && __FMA__
    Gemm_x86_fma_utility::pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
#elif NCNN_RUNTIME_CPU && __AVX__
    Gemm_x86_avx_utility::pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
#else
    Gemm_x86_utility::pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
#endif
}

static void convolution_gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

#if NCNN_RUNTIME_CPU && __AVX512F__
    Gemm_x86_avx512_utility::gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
#elif NCNN_RUNTIME_CPU && __FMA__
    Gemm_x86_fma_utility::gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
#elif NCNN_RUNTIME_CPU && __AVX__
    Gemm_x86_avx_utility::gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
#else
    Gemm_x86_utility::gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
#endif
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __AVX512F__
        int tile_size = (l2_cache_size_int8 - 64) / 16;
#elif __AVX__
        int tile_size = (l2_cache_size_int8 - 32) / 8;
#elif __SSE2__
        int tile_size = (l2_cache_size_int8 - 16) / 8;
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
#endif

#if __AVX512F__
        TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX__
        TILE_K = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __AVX512F__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 15) / 16 * 16);
#elif __AVX__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __SSE2__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __AVX512F__
        int nn_M = (M + 63) / 64;
#elif __AVX__
        int nn_M = (M + 31) / 32;
#elif __SSE2__
        int nn_M = (M + 15) / 16;
#else
        int nn_M = (M + 7) / 8;
#endif

#if __AVX512F__
        TILE_M = std::max(16, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::max(4, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __AVX512F__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

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
#endif
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M * 4 + TILE_K);
        }

#if __AVX512F__
        TILE_N = std::max(15, tile_size / 16 * 16);
#elif __AVX__
        TILE_N = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __AVX512F__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 15) / 16 * 16);
#elif __AVX__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#elif __SSE2__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;
    const int cstep = (int)bottom_blob.cstep;

    // NCNN_LOGE("convolution_im2col_input_tile_conv1x1s1d1_int8  %d %d %d %d  @%d", j, max_jj, k, max_kk, elempack);

    signed char* pp = B;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m128i _r01 = _mm_load_si128((const __m128i*)p0);
                __m128i _r23 = _mm_load_si128((const __m128i*)(p0 + 16));
                __m128i _r45 = _mm_load_si128((const __m128i*)(p0 + 32));
                __m128i _r67 = _mm_load_si128((const __m128i*)(p0 + 48));
                __m128i _r89 = _mm_load_si128((const __m128i*)(p0 + 64));
                __m128i _rab = _mm_load_si128((const __m128i*)(p0 + 80));
                __m128i _rcd = _mm_load_si128((const __m128i*)(p0 + 96));
                __m128i _ref = _mm_load_si128((const __m128i*)(p0 + 112));

#if __AVX512VNNI__ || __AVXVNNI__
                // 0011
                // 2233
                // 4455
                // 6677
                __m128i _t0 = _mm_unpacklo_epi32(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi32(_r01, _r23);
                __m128i _t2 = _mm_unpacklo_epi32(_r45, _r67);
                __m128i _t3 = _mm_unpackhi_epi32(_r45, _r67);
                __m128i _t4 = _mm_unpacklo_epi32(_r89, _rab);
                __m128i _t5 = _mm_unpackhi_epi32(_r89, _rab);
                __m128i _t6 = _mm_unpacklo_epi32(_rcd, _ref);
                __m128i _t7 = _mm_unpackhi_epi32(_rcd, _ref);

                // 0202
                // 1313
                // 4646
                // 5757
                // 8a8a
                // 9b9b
                // cece
                // dfdf
                __m128i _r0 = _mm_unpacklo_epi32(_t0, _t1);
                __m128i _r1 = _mm_unpacklo_epi32(_t2, _t3);
                __m128i _r2 = _mm_unpacklo_epi32(_t4, _t5);
                __m128i _r3 = _mm_unpacklo_epi32(_t6, _t7);
                __m128i _r4 = _mm_unpackhi_epi32(_t0, _t1);
                __m128i _r5 = _mm_unpackhi_epi32(_t2, _t3);
                __m128i _r6 = _mm_unpackhi_epi32(_t4, _t5);
                __m128i _r7 = _mm_unpackhi_epi32(_t6, _t7);

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                __m128i _v127 = _mm_set1_epi8(127);
                _r0 = _mm_add_epi8(_r0, _v127);
                _r1 = _mm_add_epi8(_r1, _v127);
                _r2 = _mm_add_epi8(_r2, _v127);
                _r3 = _mm_add_epi8(_r3, _v127);
                _r4 = _mm_add_epi8(_r4, _v127);
                _r5 = _mm_add_epi8(_r5, _v127);
                _r6 = _mm_add_epi8(_r6, _v127);
                _r7 = _mm_add_epi8(_r7, _v127);
#endif // __AVXVNNIINT8__

#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128i _t0 = _mm_unpacklo_epi16(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi16(_r01, _r23);
                __m128i _t2 = _mm_unpacklo_epi16(_r45, _r67);
                __m128i _t3 = _mm_unpackhi_epi16(_r45, _r67);
                __m128i _t4 = _mm_unpacklo_epi16(_r89, _rab);
                __m128i _t5 = _mm_unpackhi_epi16(_r89, _rab);
                __m128i _t6 = _mm_unpacklo_epi16(_rcd, _ref);
                __m128i _t7 = _mm_unpackhi_epi16(_rcd, _ref);

                _r01 = _mm_unpacklo_epi16(_t0, _t1);
                _r23 = _mm_unpackhi_epi16(_t0, _t1);
                _r45 = _mm_unpacklo_epi16(_t2, _t3);
                _r67 = _mm_unpackhi_epi16(_t2, _t3);
                _r89 = _mm_unpacklo_epi16(_t4, _t5);
                _rab = _mm_unpackhi_epi16(_t4, _t5);
                _rcd = _mm_unpacklo_epi16(_t6, _t7);
                _ref = _mm_unpackhi_epi16(_t6, _t7);

                __m128i _r0 = _mm_unpacklo_epi64(_r01, _r45);
                __m128i _r1 = _mm_unpacklo_epi64(_r89, _rcd);
                __m128i _r2 = _mm_unpackhi_epi64(_r01, _r45);
                __m128i _r3 = _mm_unpackhi_epi64(_r89, _rcd);
                __m128i _r4 = _mm_unpacklo_epi64(_r23, _r67);
                __m128i _r5 = _mm_unpacklo_epi64(_rab, _ref);
                __m128i _r6 = _mm_unpackhi_epi64(_r23, _r67);
                __m128i _r7 = _mm_unpackhi_epi64(_rab, _ref);
#endif // __AVX512VNNI__ || __AVXVNNI__

                _mm_store_si128((__m128i*)pp, _r0);
                _mm_store_si128((__m128i*)(pp + 16), _r1);
                _mm_store_si128((__m128i*)(pp + 32), _r2);
                _mm_store_si128((__m128i*)(pp + 48), _r3);
                _mm_store_si128((__m128i*)(pp + 64), _r4);
                _mm_store_si128((__m128i*)(pp + 80), _r5);
                _mm_store_si128((__m128i*)(pp + 96), _r6);
                _mm_store_si128((__m128i*)(pp + 112), _r7);

                pp += 128;
                p0 += cstep * 8;
            }
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + cstep));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + cstep * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + cstep * 3));
                // 00000000
                // 11111111
                // 22222222
                // 33333333
                transpose16x4_epi8(_r0, _r1, _r2, _r3);
#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                __m128i _v127 = _mm_set1_epi8(127);
                _r0 = _mm_add_epi8(_r0, _v127);
                _r1 = _mm_add_epi8(_r1, _v127);
                _r2 = _mm_add_epi8(_r2, _v127);
                _r3 = _mm_add_epi8(_r3, _v127);
#endif // __AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 16), _r1);
                _mm_storeu_si128((__m128i*)(pp + 32), _r2);
                _mm_storeu_si128((__m128i*)(pp + 48), _r3);
                pp += 64;
                p0 += cstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + cstep));
                __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                __m128i _r23 = _mm_unpackhi_epi8(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _r01);
                _mm_storeu_si128((__m128i*)(pp + 16), _r23);
                pp += 32;
                p0 += cstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 16;
                p0 += cstep;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m128i _r01 = _mm_load_si128((const __m128i*)p0);
                __m128i _r23 = _mm_load_si128((const __m128i*)(p0 + 16));
                __m128i _r45 = _mm_load_si128((const __m128i*)(p0 + 32));
                __m128i _r67 = _mm_load_si128((const __m128i*)(p0 + 48));

#if __AVX512VNNI__ || __AVXVNNI__
                // 0011
                // 2233
                // 4455
                // 6677
                __m128i _t0 = _mm_unpacklo_epi32(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi32(_r01, _r23);
                __m128i _t2 = _mm_unpacklo_epi32(_r45, _r67);
                __m128i _t3 = _mm_unpackhi_epi32(_r45, _r67);

                // 0202
                // 1313
                // 4646
                // 5757
                __m128i _r0 = _mm_unpacklo_epi32(_t0, _t1);
                __m128i _r1 = _mm_unpacklo_epi32(_t2, _t3);
                __m128i _r2 = _mm_unpackhi_epi32(_t0, _t1);
                __m128i _r3 = _mm_unpackhi_epi32(_t2, _t3);

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                __m128i _v127 = _mm_set1_epi8(127);
                _r0 = _mm_add_epi8(_r0, _v127);
                _r1 = _mm_add_epi8(_r1, _v127);
                _r2 = _mm_add_epi8(_r2, _v127);
                _r3 = _mm_add_epi8(_r3, _v127);
#endif // __AVXVNNIINT8__

#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128i _t0 = _mm_unpacklo_epi16(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi16(_r01, _r23);
                __m128i _t2 = _mm_unpacklo_epi16(_r45, _r67);
                __m128i _t3 = _mm_unpackhi_epi16(_r45, _r67);

                _r01 = _mm_unpacklo_epi16(_t0, _t1);
                _r23 = _mm_unpackhi_epi16(_t0, _t1);
                _r45 = _mm_unpacklo_epi16(_t2, _t3);
                _r67 = _mm_unpackhi_epi16(_t2, _t3);

                __m128i _r0 = _mm_unpacklo_epi64(_r01, _r45);
                __m128i _r1 = _mm_unpackhi_epi64(_r01, _r45);
                __m128i _r2 = _mm_unpacklo_epi64(_r23, _r67);
                __m128i _r3 = _mm_unpackhi_epi64(_r23, _r67);
#endif // __AVX512VNNI__ || __AVXVNNI__

                _mm_store_si128((__m128i*)pp, _r0);
                _mm_store_si128((__m128i*)(pp + 16), _r1);
                _mm_store_si128((__m128i*)(pp + 32), _r2);
                _mm_store_si128((__m128i*)(pp + 48), _r3);

                pp += 64;
                p0 += cstep * 8;
            }
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + cstep));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + cstep * 2));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + cstep * 3));
                // 0000....
                // 1111....
                // 2222....
                // 3333....
                __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                __m128i _r23 = _mm_unpacklo_epi8(_r2, _r3);
                // 01010101
                // 23232323
                _r0 = _mm_unpacklo_epi16(_r01, _r23);
                _r1 = _mm_unpackhi_epi16(_r01, _r23);
#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                __m128i _v127 = _mm_set1_epi8(127);
                _r0 = _mm_add_epi8(_r0, _v127);
                _r1 = _mm_add_epi8(_r1, _v127);
#endif // __AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 16), _r1);
                pp += 32;
                p0 += cstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + cstep));
                __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _r01);
                pp += 16;
                p0 += cstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 8;
                p0 += cstep;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m128i _r01 = _mm_load_si128((const __m128i*)p0);
                __m128i _r23 = _mm_load_si128((const __m128i*)(p0 + 16));

#if __AVX512VNNI__ || __AVXVNNI__
                __m128i _t0 = _mm_unpacklo_epi32(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi32(_r01, _r23);

                __m128i _r0 = _mm_unpacklo_epi32(_t0, _t1);
                __m128i _r1 = _mm_unpackhi_epi32(_t0, _t1);

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                __m128i _v127 = _mm_set1_epi8(127);
                _r0 = _mm_add_epi8(_r0, _v127);
                _r1 = _mm_add_epi8(_r1, _v127);
#endif // __AVXVNNIINT8__

#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128i _t0 = _mm_unpacklo_epi16(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi16(_r01, _r23);

                __m128i _r0 = _mm_unpacklo_epi16(_t0, _t1);
                __m128i _r1 = _mm_unpackhi_epi16(_t0, _t1);
#endif // __AVX512VNNI__ || __AVXVNNI__

                _mm_store_si128((__m128i*)pp, _r0);
                _mm_store_si128((__m128i*)(pp + 16), _r1);

                pp += 32;
                p0 += cstep * 8;
            }
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVXVNNIINT8__
                pp[0] = p0[0];
                pp[1] = p0[cstep];
                pp[2] = p0[cstep * 2];
                pp[3] = p0[cstep * 3];
                pp[4] = p0[1];
                pp[5] = p0[cstep + 1];
                pp[6] = p0[cstep * 2 + 1];
                pp[7] = p0[cstep * 3 + 1];
                pp[8] = p0[2];
                pp[9] = p0[cstep + 2];
                pp[10] = p0[cstep * 2 + 2];
                pp[11] = p0[cstep * 3 + 2];
                pp[12] = p0[3];
                pp[13] = p0[cstep + 3];
                pp[14] = p0[cstep * 2 + 3];
                pp[15] = p0[cstep * 3 + 3];
#else  // __AVXVNNIINT8__
                pp[0] = p0[0] + 127;
                pp[1] = p0[cstep] + 127;
                pp[2] = p0[cstep * 2] + 127;
                pp[3] = p0[cstep * 3] + 127;
                pp[4] = p0[1] + 127;
                pp[5] = p0[cstep + 1] + 127;
                pp[6] = p0[cstep * 2 + 1] + 127;
                pp[7] = p0[cstep * 3 + 1] + 127;
                pp[8] = p0[2] + 127;
                pp[9] = p0[cstep + 2] + 127;
                pp[10] = p0[cstep * 2 + 2] + 127;
                pp[11] = p0[cstep * 3 + 2] + 127;
                pp[12] = p0[3] + 127;
                pp[13] = p0[cstep + 3] + 127;
                pp[14] = p0[cstep * 2 + 3] + 127;
                pp[15] = p0[cstep * 3 + 3] + 127;
#endif // __AVXVNNIINT8__
                pp += 16;
                p0 += cstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep + 0];
                pp[2] = p0[1];
                pp[3] = p0[cstep + 1];
                pp[4] = p0[2];
                pp[5] = p0[cstep + 2];
                pp[6] = p0[3];
                pp[7] = p0[cstep + 3];
                pp += 8;
                p0 += cstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += cstep;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __SSE2__
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + 8));
#if __AVX512VNNI__ || __AVXVNNI__
                __m128i _r01 = _mm_unpacklo_epi32(_r0, _r1);
#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                _r01 = _mm_add_epi8(_r01, _mm_set1_epi8(127));
#endif // __AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
#endif // __AVX512VNNI__ || __AVXVNNI__
                _mm_storeu_si128((__m128i*)pp, _r01);
                pp += 16;
                p0 += cstep * 8;
            }
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVXVNNIINT8__
                pp[0] = p0[0];
                pp[1] = p0[cstep];
                pp[2] = p0[cstep * 2];
                pp[3] = p0[cstep * 3];
                pp[4] = p0[1];
                pp[5] = p0[cstep + 1];
                pp[6] = p0[cstep * 2 + 1];
                pp[7] = p0[cstep * 3 + 1];
#else  // __AVXVNNIINT8__
                pp[0] = p0[0] + 127;
                pp[1] = p0[cstep] + 127;
                pp[2] = p0[cstep * 2] + 127;
                pp[3] = p0[cstep * 3] + 127;
                pp[4] = p0[1] + 127;
                pp[5] = p0[cstep + 1] + 127;
                pp[6] = p0[cstep * 2 + 1] + 127;
                pp[7] = p0[cstep * 3 + 1] + 127;
#endif // __AVXVNNIINT8__
                pp += 8;
                p0 += cstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep];
                pp[2] = p0[1];
                pp[3] = p0[cstep + 1];
                pp += 4;
                p0 += cstep * 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __SSE2__
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                __m128i _v127 = _mm_set1_epi8(127);
                _r0 = _mm_add_epi8(_r0, _v127);
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                _mm_storel_epi64((__m128i*)pp, _r0);
                pp += 8;
                p0 += cstep * 8;
            }
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVXVNNIINT8__
                pp[0] = p0[0];
                pp[1] = p0[cstep];
                pp[2] = p0[cstep * 2];
                pp[3] = p0[cstep * 3];
#else  // __AVXVNNIINT8__
                pp[0] = p0[0] + 127;
                pp[1] = p0[cstep] + 127;
                pp[2] = p0[cstep * 2] + 127;
                pp[3] = p0[cstep * 3] + 127;
#endif // __AVXVNNIINT8__
                pp += 4;
                p0 += cstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += cstep;
            }
        }
    }
}

static void convolution_im2col_input_tile_int8_impl(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int cstep = (int)bottom_blob.cstep;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    // NCNN_LOGE("convolution_im2col_input_tile_int8_impl %d %d %d %d   %d  @%d", j, max_jj, k, max_kk, maxk, elempack);

    signed char* pp = B;

#if __SSE2__
    FastDivider_epu32 div_outw(outw);
    FastDivider_epu32 div_maxk(maxk);
    FastDivider_epu32 div_kernel_w(kernel_w);
#endif

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        __m512i _dy;
        __m512i _dx;
        {
            __m512i _offset = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i _dxy = _mm512_add_epi32(_mm512_set1_epi32(j + jj), _offset);
            _dy = div_outw._mm512_comp_div_epu32(_dxy);
            _dx = _mm512_sub_epi32(_dxy, _mm512_mullo_epi32(_dy, _mm512_set1_epi32(outw)));
            _dy = _mm512_mullo_epi32(_dy, _mm512_set1_epi32(stride_h));
            _dx = _mm512_mullo_epi32(_dx, _mm512_set1_epi32(stride_w));
            _dy = _mm512_mullo_epi32(_dy, _mm512_set1_epi32(w));
        }

        __m512i _dxy_offset = _mm512_add_epi32(_dx, _dy);

        const int dy0 = _mm_extract_epi32(_mm512_extracti32x4_epi32(_dy, 0), 0);
        const int dyf = _mm_extract_epi32(_mm512_extracti32x4_epi32(_dy, 3), 3);

        if (dy0 == dyf && stride_w == 1)
        {
            const int dx0 = _mm_extract_epi32(_mm512_extracti32x4_epi32(_dx, 0), 0);

            const int dxy_offset = dx0 + dy0;

            if (elempack == 1)
            {
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _puv_offset;
                    {
                        __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                        __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                        __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                        __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                        __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                        _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                        _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                        _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                    }

                    __m128i _offset = _mm_add_epi32(_mm_set1_epi32(dxy_offset), _puv_offset);

                    __m128i _p0 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + _mm_extract_epi32(_offset, 0)));
                    __m128i _p1 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + _mm_extract_epi32(_offset, 1)));
                    __m128i _p2 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + _mm_extract_epi32(_offset, 2)));
                    __m128i _p3 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + _mm_extract_epi32(_offset, 3)));

                    transpose16x4_epi8(_p0, _p1, _p2, _p3);

#if __AVXVNNIINT8__
#else
                    __m128i _v127 = _mm_set1_epi8(127);
                    _p0 = _mm_add_epi8(_p0, _v127);
                    _p1 = _mm_add_epi8(_p1, _v127);
                    _p2 = _mm_add_epi8(_p2, _v127);
                    _p3 = _mm_add_epi8(_p3, _v127);
#endif

                    _mm_store_si128((__m128i*)pp, _p0);
                    _mm_store_si128((__m128i*)(pp + 16), _p1);
                    _mm_store_si128((__m128i*)(pp + 32), _p2);
                    _mm_store_si128((__m128i*)(pp + 48), _p3);

                    pp += 64;
                }
#endif //__AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                    int offset1 = dxy_offset + p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

                    __m128i _p0 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset0));
                    __m128i _p1 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset1));

                    __m128i _t0 = _mm_unpacklo_epi8(_p0, _p1);
                    __m128i _t1 = _mm_unpackhi_epi8(_p0, _p1);
                    _mm_store_si128((__m128i*)pp, _t0);
                    _mm_store_si128((__m128i*)(pp + 16), _t1);

                    pp += 32;
                }
                for (; kk < max_kk; kk += 1)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;

                    int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;

                    __m128i _p0 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset0));

                    _mm_store_si128((__m128i*)pp, _p0);

                    pp += 16;
                }
            }
            if (elempack == 8)
            {
                int kk = 0;
                for (; kk < max_kk / 8; kk++)
                {
                    int p = (k / 8 + kk) / maxk;
                    int uv = (k / 8 + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int offset = (dxy_offset + p * cstep + u * dilation_h * w + v * dilation_w) * 8;

                    __m512i _r0 = _mm512_loadu_si512((const __m512i*)((const signed char*)bottom_blob + offset));
                    __m512i _r1 = _mm512_loadu_si512((const __m512i*)((const signed char*)bottom_blob + offset + 64));

#if __AVX512VNNI__ || __AVXVNNI__

                    // 0011 2233 4455 6677
                    // 8899 aabb ccdd eeff

                    __m512i _r2 = _mm512_unpacklo_epi32(_r0, _r1);
                    __m512i _r3 = _mm512_unpackhi_epi32(_r0, _r1);

                    // 0808 2a2a 4c4c 6e6e
                    // 1919 3b3b 5d5d 7f7f

                    _r0 = _mm512_unpacklo_epi32(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi32(_r2, _r3);

                    // 0189 23ab 45cd 67ef
                    // 0189 23ab 45cd 67ef

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 1, 3, 1));

                    // 0189 45cd 0189 45cd
                    // 23ab 67ef 23ab 67ef

                    _r0 = _mm512_unpacklo_epi64(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi64(_r2, _r3);

                    // 0123 4567 0123 4567
                    // 89ab cdef 89ab cdef

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(1, 0, 1, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 2, 3, 2));

                    _r0 = _r2;
                    _r1 = _r3;

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    __m512i _v127 = _mm512_set1_epi8(127);
                    _r0 = _mm512_add_epi8(_r0, _v127);
                    _r1 = _mm512_add_epi8(_r1, _v127);
#endif // __AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__

                    // 00001111 22223333 44445555 66667777
                    // 88889999 aaaabbbb ccccdddd eeeeffff

                    __m512i _r2 = _mm512_unpacklo_epi16(_r0, _r1);
                    __m512i _r3 = _mm512_unpackhi_epi16(_r0, _r1);

                    // 08080808 2a2a2a2a 4c4c4c4c 6e6e6e6e
                    // 19191919 3b3b3b3b 5d5d5d5d 7f7f7f7f

                    _r0 = _mm512_unpacklo_epi16(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi16(_r2, _r3);

                    // 01890189 23ab23ab 45cd45cd 67ef67ef
                    // 01890189 23ab23ab 45cd45cd 67ef67ef

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 1, 3, 1));

                    // 01890189 45cd45cd 01890189 45cd45cd
                    // 23ab23ab 67ef67ef 23ab23ab 67ef67ef

                    _r0 = _mm512_unpacklo_epi32(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi32(_r2, _r3);

                    // 012389ab 4567cdef 012389ab 4567cdef      0 0 2 2
                    // 012389ab 4567cdef 012389ab 4567cdef      1 1 3 3

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 1, 3, 1));

                    // 012389ab 012389ab 012389ab 012389ab      0 2 1 3
                    // 4567cdef 4567cdef 4567cdef 4567cdef      0 2 1 3

                    _r0 = _mm512_unpacklo_epi64(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi64(_r2, _r3);

                    // 01234567 01234567 01234567 01234567      0 2 1 3
                    // 89abcdef 89abcdef 89abcdef 89abcdef      0 2 1 3

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(1, 0, 1, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 2, 3, 2));

                    // 01234567 01234567 89abcdef 89abcdef      0 2 0 2
                    // 01234567 01234567 89abcdef 89abcdef      1 3 1 3

                    _r0 = _mm512_shuffle_i32x4(_r2, _r3, _MM_SHUFFLE(2, 0, 2, 0));
                    _r1 = _mm512_shuffle_i32x4(_r2, _r3, _MM_SHUFFLE(3, 1, 3, 1));

#endif // __AVX512VNNI__ || __AVXVNNI__
                    _mm512_storeu_si512((__m512i*)pp, _r0);
                    _mm512_storeu_si512((__m512i*)(pp + 64), _r1);
                    pp += 128;
                }
            }
        }
        else
        {
            if (elempack == 1)
            {
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _puv_offset;
                    {
                        __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                        __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                        __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                        __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                        __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                        _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                        _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                        _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                    }

                    __m512i _vindex0 = _mm512_add_epi32(_dxy_offset, _mm512_set1_epi32(_mm_extract_epi32(_puv_offset, 0)));
                    __m512i _vindex1 = _mm512_add_epi32(_dxy_offset, _mm512_set1_epi32(_mm_extract_epi32(_puv_offset, 1)));
                    __m512i _vindex2 = _mm512_add_epi32(_dxy_offset, _mm512_set1_epi32(_mm_extract_epi32(_puv_offset, 2)));
                    __m512i _vindex3 = _mm512_add_epi32(_dxy_offset, _mm512_set1_epi32(_mm_extract_epi32(_puv_offset, 3)));

                    __m128i _p0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex0, bottom_blob, sizeof(signed char)));
                    __m128i _p1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex1, bottom_blob, sizeof(signed char)));
                    __m128i _p2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex2, bottom_blob, sizeof(signed char)));
                    __m128i _p3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex3, bottom_blob, sizeof(signed char)));

                    transpose16x4_epi8(_p0, _p1, _p2, _p3);

#if __AVXVNNIINT8__
#else
                    __m128i _v127 = _mm_set1_epi8(127);
                    _p0 = _mm_add_epi8(_p0, _v127);
                    _p1 = _mm_add_epi8(_p1, _v127);
                    _p2 = _mm_add_epi8(_p2, _v127);
                    _p3 = _mm_add_epi8(_p3, _v127);
#endif

                    _mm_store_si128((__m128i*)pp, _p0);
                    _mm_store_si128((__m128i*)(pp + 16), _p1);
                    _mm_store_si128((__m128i*)(pp + 32), _p2);
                    _mm_store_si128((__m128i*)(pp + 48), _p3);

                    pp += 64;
                }
#endif //__AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    int puv_offset0 = p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                    int puv_offset1 = p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

                    __m512i _vindex0 = _mm512_add_epi32(_dxy_offset, _mm512_set1_epi32(puv_offset0));
                    __m512i _vindex1 = _mm512_add_epi32(_dxy_offset, _mm512_set1_epi32(puv_offset1));

                    __m128i _p0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex0, bottom_blob, sizeof(signed char)));
                    __m128i _p1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex1, bottom_blob, sizeof(signed char)));

                    __m128i _t0 = _mm_unpacklo_epi8(_p0, _p1);
                    __m128i _t1 = _mm_unpackhi_epi8(_p0, _p1);
                    _mm_store_si128((__m128i*)pp, _t0);
                    _mm_store_si128((__m128i*)(pp + 16), _t1);

                    pp += 32;
                }
                for (; kk < max_kk; kk += 1)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;

                    int puv_offset0 = p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;

                    __m512i _vindex0 = _mm512_add_epi32(_dxy_offset, _mm512_set1_epi32(puv_offset0));

                    __m128i _p0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex0, bottom_blob, sizeof(signed char)));

                    _mm_store_si128((__m128i*)pp, _p0);

                    pp += 16;
                }
            }
            if (elempack == 8)
            {
                int kk = 0;
                for (; kk < max_kk / 8; kk++)
                {
                    int p = (k / 8 + kk) / maxk;
                    int uv = (k / 8 + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int puv_offset = p * cstep + u * dilation_h * w + v * dilation_w;

                    __m512i _vindex = _mm512_add_epi32(_dxy_offset, _mm512_set1_epi32(puv_offset));

                    _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(8));

                    __m512i _r0 = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(_vindex, 0), bottom_blob, sizeof(signed char));
                    __m512i _r1 = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(_vindex, 1), bottom_blob, sizeof(signed char));

#if __AVX512VNNI__ || __AVXVNNI__

                    // 0011 2233 4455 6677
                    // 8899 aabb ccdd eeff

                    __m512i _r2 = _mm512_unpacklo_epi32(_r0, _r1);
                    __m512i _r3 = _mm512_unpackhi_epi32(_r0, _r1);

                    // 0808 2a2a 4c4c 6e6e
                    // 1919 3b3b 5d5d 7f7f

                    _r0 = _mm512_unpacklo_epi32(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi32(_r2, _r3);

                    // 0189 23ab 45cd 67ef
                    // 0189 23ab 45cd 67ef

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 1, 3, 1));

                    // 0189 45cd 0189 45cd
                    // 23ab 67ef 23ab 67ef

                    _r0 = _mm512_unpacklo_epi64(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi64(_r2, _r3);

                    // 0123 4567 0123 4567
                    // 89ab cdef 89ab cdef

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(1, 0, 1, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 2, 3, 2));

                    _r0 = _r2;
                    _r1 = _r3;

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    __m512i _v127 = _mm512_set1_epi8(127);
                    _r0 = _mm512_add_epi8(_r0, _v127);
                    _r1 = _mm512_add_epi8(_r1, _v127);
#endif // __AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__

                    // 00001111 22223333 44445555 66667777
                    // 88889999 aaaabbbb ccccdddd eeeeffff

                    __m512i _r2 = _mm512_unpacklo_epi16(_r0, _r1);
                    __m512i _r3 = _mm512_unpackhi_epi16(_r0, _r1);

                    // 08080808 2a2a2a2a 4c4c4c4c 6e6e6e6e
                    // 19191919 3b3b3b3b 5d5d5d5d 7f7f7f7f

                    _r0 = _mm512_unpacklo_epi16(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi16(_r2, _r3);

                    // 01890189 23ab23ab 45cd45cd 67ef67ef
                    // 01890189 23ab23ab 45cd45cd 67ef67ef

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 1, 3, 1));

                    // 01890189 45cd45cd 01890189 45cd45cd
                    // 23ab23ab 67ef67ef 23ab23ab 67ef67ef

                    _r0 = _mm512_unpacklo_epi32(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi32(_r2, _r3);

                    // 012389ab 4567cdef 012389ab 4567cdef      0 0 2 2
                    // 012389ab 4567cdef 012389ab 4567cdef      1 1 3 3

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 1, 3, 1));

                    // 012389ab 012389ab 012389ab 012389ab      0 2 1 3
                    // 4567cdef 4567cdef 4567cdef 4567cdef      0 2 1 3

                    _r0 = _mm512_unpacklo_epi64(_r2, _r3);
                    _r1 = _mm512_unpackhi_epi64(_r2, _r3);

                    // 01234567 01234567 01234567 01234567      0 2 1 3
                    // 89abcdef 89abcdef 89abcdef 89abcdef      0 2 1 3

                    _r2 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(1, 0, 1, 0));
                    _r3 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 2, 3, 2));

                    // 01234567 01234567 89abcdef 89abcdef      0 2 0 2
                    // 01234567 01234567 89abcdef 89abcdef      1 3 1 3

                    _r0 = _mm512_shuffle_i32x4(_r2, _r3, _MM_SHUFFLE(2, 0, 2, 0));
                    _r1 = _mm512_shuffle_i32x4(_r2, _r3, _MM_SHUFFLE(3, 1, 3, 1));

#endif // __AVX512VNNI__ || __AVXVNNI__
                    _mm512_storeu_si512((__m512i*)pp, _r0);
                    _mm512_storeu_si512((__m512i*)(pp + 64), _r1);
                    pp += 128;
                }
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __AVX2__
        __m256i _dy;
        __m256i _dx;
        {
            __m256i _offset = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i _dxy = _mm256_add_epi32(_mm256_set1_epi32(j + jj), _offset);
            _dy = div_outw._mm256_comp_div_epu32(_dxy);
            _dx = _mm256_sub_epi32(_dxy, _mm256_mullo_epi32(_dy, _mm256_set1_epi32(outw)));
            _dy = _mm256_mullo_epi32(_dy, _mm256_set1_epi32(stride_h));
            _dx = _mm256_mullo_epi32(_dx, _mm256_set1_epi32(stride_w));
            _dy = _mm256_mullo_epi32(_dy, _mm256_set1_epi32(w));
        }

        __m256i _dxy_offset = _mm256_add_epi32(_dx, _dy);
#else // __AVX2__

        __m128i _dy0;
        __m128i _dy1;
        __m128i _dx0;
        __m128i _dx1;
        {
            __m128i _offset0 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i _dxy0 = _mm_add_epi32(_mm_set1_epi32(j + jj), _offset0);
            __m128i _dxy1 = _mm_add_epi32(_dxy0, _mm_set1_epi32(4));
            _dy0 = div_outw._mm_comp_div_epu32(_dxy0);
            _dy1 = div_outw._mm_comp_div_epu32(_dxy1);
            _dx0 = _mm_sub_epi32(_dxy0, _mm_comp_mullo_epi32(_dy0, _mm_set1_epi32(outw)));
            _dx1 = _mm_sub_epi32(_dxy1, _mm_comp_mullo_epi32(_dy1, _mm_set1_epi32(outw)));
            _dy0 = _mm_comp_mullo_epi32(_dy0, _mm_set1_epi32(stride_h));
            _dy1 = _mm_comp_mullo_epi32(_dy1, _mm_set1_epi32(stride_h));
            _dx0 = _mm_comp_mullo_epi32(_dx0, _mm_set1_epi32(stride_w));
            _dx1 = _mm_comp_mullo_epi32(_dx1, _mm_set1_epi32(stride_w));
            _dy0 = _mm_comp_mullo_epi32(_dy0, _mm_set1_epi32(w));
            _dy1 = _mm_comp_mullo_epi32(_dy1, _mm_set1_epi32(w));
        }

        __m128i _dxy_offset0 = _mm_add_epi32(_dx0, _dy0);
        __m128i _dxy_offset1 = _mm_add_epi32(_dx1, _dy1);

#endif // __AVX2__

#if __AVX2__
        const int dy0 = _mm_extract_epi32(_mm256_extractf128_si256(_dy, 0), 0);
        const int dy7 = _mm_extract_epi32(_mm256_extractf128_si256(_dy, 1), 3);
#else
        const int dy0 = _mm_cvtsi128_si32(_dy0);
        const int dy7 = _mm_cvtsi128_si32(_mm_shuffle_epi32(_dy1, _MM_SHUFFLE(3, 3, 3, 3)));
#endif

        if (dy0 == dy7 && stride_w == 1)
        {
#if __AVX2__
            const int dx0 = _mm_extract_epi32(_mm256_extractf128_si256(_dx, 0), 0);
#else
            const int dx0 = _mm_cvtsi128_si32(_dx0);
#endif

            const int dxy_offset = dx0 + dy0;

            if (elempack == 1)
            {
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _puv_offset;
                    {
                        __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                        __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                        __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                        __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                        __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                        _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                        _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                        _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                    }

                    __m128i _offset = _mm_add_epi32(_mm_set1_epi32(dxy_offset), _puv_offset);

                    __m256i _p0 = _mm256_i32gather_epi64((const long long int*)bottom_blob, _offset, sizeof(signed char));

                    // 0000000011111111 2222222233333333

                    _p0 = _mm256_shuffle_epi32(_p0, _MM_SHUFFLE(3, 1, 2, 0));

                    // 0000111100001111 2222333322223333

                    _p0 = _mm256_permute4x64_epi64(_p0, _MM_SHUFFLE(3, 1, 2, 0));

                    // 0000111122223333 0000111122223333

                    _p0 = _mm256_shuffle_epi8(_p0, _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));

#if __AVXVNNIINT8__
#else
                    __m256i _v127 = _mm256_set1_epi8(127);
                    _p0 = _mm256_add_epi8(_p0, _v127);
#endif

                    _mm256_storeu_si256((__m256i*)pp, _p0);
                    pp += 32;
                }
#endif //__AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                    int offset1 = dxy_offset + p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

                    __m128i _p0 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offset0));
                    __m128i _p1 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offset1));

                    __m128i _r01 = _mm_unpacklo_epi8(_p0, _p1);
                    _mm_storeu_si128((__m128i*)pp, _r01);
                    pp += 16;
                }
                for (; kk < max_kk; kk += 1)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;

                    int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;

                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offset0));
                    _mm_storel_epi64((__m128i*)pp, _r0);
                    pp += 8;
                }
            }
            if (elempack == 8)
            {
                int kk = 0;
                for (; kk < max_kk / 8; kk++)
                {
                    int p = (k / 8 + kk) / maxk;
                    int uv = (k / 8 + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int offset = (dxy_offset + p * cstep + u * dilation_h * w + v * dilation_w) * 8;

#if __AVX2__
                    __m256i _r0 = _mm256_loadu_si256((const __m256i*)((const signed char*)bottom_blob + offset));
                    __m256i _r1 = _mm256_loadu_si256((const __m256i*)((const signed char*)bottom_blob + offset + 32));

#if __AVX512VNNI__ || __AVXVNNI__

                    // 0011 2233
                    // 4455 6677

                    __m256i _r2 = _mm256_unpacklo_epi32(_r0, _r1);
                    __m256i _r3 = _mm256_unpackhi_epi32(_r0, _r1);

                    // 0404 2626
                    // 1515 3737

                    _r0 = _mm256_unpacklo_epi32(_r2, _r3);
                    _r1 = _mm256_unpackhi_epi32(_r2, _r3);

                    // 0145 2367
                    // 0145 2367

                    _r0 = _mm256_permute4x64_epi64(_r0, _MM_SHUFFLE(3, 1, 2, 0));
                    _r1 = _mm256_permute4x64_epi64(_r1, _MM_SHUFFLE(3, 1, 2, 0));

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    __m256i _v127 = _mm256_set1_epi8(127);
                    _r0 = _mm256_add_epi8(_r0, _v127);
                    _r1 = _mm256_add_epi8(_r1, _v127);
#endif // __AVXVNNIINT8__

#else // __AVX512VNNI__ || __AVXVNNI__

                    // 00001111 22223333
                    // 44445555 66667777

                    __m256i _r2 = _mm256_unpacklo_epi16(_r0, _r1);
                    __m256i _r3 = _mm256_unpackhi_epi16(_r0, _r1);

                    // 04040404 26262626
                    // 15151515 37373737

                    _r0 = _mm256_unpacklo_epi16(_r2, _r3);
                    _r1 = _mm256_unpackhi_epi16(_r2, _r3);

                    // 01450145 23672367
                    // 01450145 23672367

                    _r0 = _mm256_permute4x64_epi64(_r0, _MM_SHUFFLE(3, 1, 2, 0));
                    _r1 = _mm256_permute4x64_epi64(_r1, _MM_SHUFFLE(3, 1, 2, 0));

                    // 01452367 01452367
                    // 01452367 01452367

                    _r0 = _mm256_shuffle_epi32(_r0, _MM_SHUFFLE(3, 1, 2, 0));
                    _r1 = _mm256_shuffle_epi32(_r1, _MM_SHUFFLE(3, 1, 2, 0));

#endif // __AVX512VNNI__ || __AVXVNNI__

                    _mm256_storeu_si256((__m256i*)pp, _r0);
                    _mm256_storeu_si256((__m256i*)(pp + 32), _r1);

#else // __AVX2__
                    __m128i _r0 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset));
                    __m128i _r1 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset + 16));
                    __m128i _r2 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset + 32));
                    __m128i _r3 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset + 48));

                    // 00001111
                    // 22223333
                    // 44445555
                    // 66667777

                    __m128i _r4 = _mm_unpacklo_epi16(_r0, _r1);
                    __m128i _r5 = _mm_unpackhi_epi16(_r0, _r1);
                    __m128i _r6 = _mm_unpacklo_epi16(_r2, _r3);
                    __m128i _r7 = _mm_unpackhi_epi16(_r2, _r3);

                    // 02020202
                    // 13131313
                    // 46464646
                    // 57575757

                    _r0 = _mm_unpacklo_epi16(_r4, _r5);
                    _r1 = _mm_unpacklo_epi16(_r6, _r7);
                    _r2 = _mm_unpackhi_epi16(_r4, _r5);
                    _r3 = _mm_unpackhi_epi16(_r6, _r7);

                    // 01230123
                    // 45674567
                    // 01230123
                    // 45674567

                    _r4 = _mm_unpacklo_epi64(_r0, _r1);
                    _r5 = _mm_unpackhi_epi64(_r0, _r1);
                    _r6 = _mm_unpacklo_epi64(_r2, _r3);
                    _r7 = _mm_unpackhi_epi64(_r2, _r3);

                    _mm_storeu_si128((__m128i*)pp, _r4);
                    _mm_storeu_si128((__m128i*)(pp + 16), _r5);
                    _mm_storeu_si128((__m128i*)(pp + 32), _r6);
                    _mm_storeu_si128((__m128i*)(pp + 48), _r7);

#endif // __AVX2__

                    pp += 64;
                }
            }
        }
        else
        {
            if (elempack == 1)
            {
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _puv_offset;
                    {
                        __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                        __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                        __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                        __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                        __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                        _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                        _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                        _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                    }

                    __m256i _vindex0 = _mm256_add_epi32(_dxy_offset, _mm256_set1_epi32(_mm_extract_epi32(_puv_offset, 0)));
                    __m256i _vindex1 = _mm256_add_epi32(_dxy_offset, _mm256_set1_epi32(_mm_extract_epi32(_puv_offset, 1)));
                    __m256i _vindex2 = _mm256_add_epi32(_dxy_offset, _mm256_set1_epi32(_mm_extract_epi32(_puv_offset, 2)));
                    __m256i _vindex3 = _mm256_add_epi32(_dxy_offset, _mm256_set1_epi32(_mm_extract_epi32(_puv_offset, 3)));

                    __m128i _p0 = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)bottom_blob, _vindex0, sizeof(signed char)));
                    __m128i _p1 = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)bottom_blob, _vindex1, sizeof(signed char)));
                    __m128i _p2 = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)bottom_blob, _vindex2, sizeof(signed char)));
                    __m128i _p3 = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)bottom_blob, _vindex3, sizeof(signed char)));

                    // 00000000........
                    // 11111111........
                    // 22222222........
                    // 33333333........

                    __m128i _p01 = _mm_unpacklo_epi8(_p0, _p1);
                    __m128i _p23 = _mm_unpacklo_epi8(_p2, _p3);

                    // 0101010101010101
                    // 2323232323232323

                    _p0 = _mm_unpacklo_epi16(_p01, _p23);
                    _p1 = _mm_unpackhi_epi16(_p01, _p23);

#if __AVXVNNIINT8__
#else // __AVXVNNIINT8__

                    __m128i _v127 = _mm_set1_epi8(127);
                    _p0 = _mm_add_epi8(_p0, _v127);
                    _p1 = _mm_add_epi8(_p1, _v127);

#endif // __AVXVNNIINT8__

                    _mm_storeu_si128((__m128i*)pp, _p0);
                    _mm_storeu_si128((__m128i*)(pp + 16), _p1);

                    pp += 32;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    int puv_offset0 = p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                    int puv_offset1 = p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

#if __AVX2__
                    __m256i _vindex0 = _mm256_add_epi32(_dxy_offset, _mm256_set1_epi32(puv_offset0));
                    __m256i _vindex1 = _mm256_add_epi32(_dxy_offset, _mm256_set1_epi32(puv_offset1));

                    __m128i _p0 = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)bottom_blob, _vindex0, sizeof(signed char)));
                    __m128i _p1 = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)bottom_blob, _vindex1, sizeof(signed char)));

                    __m128i _p01 = _mm_unpacklo_epi8(_p0, _p1);

                    _mm_storeu_si128((__m128i*)pp, _p01);

#else // __AVX2__

                    __m128i _vindex0 = _mm_add_epi32(_dxy_offset0, _mm_set1_epi32(puv_offset0));
                    __m128i _vindex1 = _mm_add_epi32(_dxy_offset1, _mm_set1_epi32(puv_offset0));
                    __m128i _vindex2 = _mm_add_epi32(_dxy_offset0, _mm_set1_epi32(puv_offset1));
                    __m128i _vindex3 = _mm_add_epi32(_dxy_offset1, _mm_set1_epi32(puv_offset1));

                    int offsets0[4];
                    int offsets1[4];
                    int offsets2[4];
                    int offsets3[4];
                    _mm_storeu_si128((__m128i*)offsets0, _vindex0);
                    _mm_storeu_si128((__m128i*)offsets1, _vindex1);
                    _mm_storeu_si128((__m128i*)offsets2, _vindex2);
                    _mm_storeu_si128((__m128i*)offsets3, _vindex3);

                    pp[0] = ((const signed char*)bottom_blob)[offsets0[0]];
                    pp[1] = ((const signed char*)bottom_blob)[offsets2[0]];
                    pp[2] = ((const signed char*)bottom_blob)[offsets0[1]];
                    pp[3] = ((const signed char*)bottom_blob)[offsets2[1]];
                    pp[4] = ((const signed char*)bottom_blob)[offsets0[2]];
                    pp[5] = ((const signed char*)bottom_blob)[offsets2[2]];
                    pp[6] = ((const signed char*)bottom_blob)[offsets0[3]];
                    pp[7] = ((const signed char*)bottom_blob)[offsets2[3]];
                    pp[8] = ((const signed char*)bottom_blob)[offsets1[0]];
                    pp[9] = ((const signed char*)bottom_blob)[offsets3[0]];
                    pp[10] = ((const signed char*)bottom_blob)[offsets1[1]];
                    pp[11] = ((const signed char*)bottom_blob)[offsets3[1]];
                    pp[12] = ((const signed char*)bottom_blob)[offsets1[2]];
                    pp[13] = ((const signed char*)bottom_blob)[offsets3[2]];
                    pp[14] = ((const signed char*)bottom_blob)[offsets1[3]];
                    pp[15] = ((const signed char*)bottom_blob)[offsets3[3]];

#endif // __AVX2__

                    pp += 16;
                }
                for (; kk < max_kk; kk += 1)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;

                    int puv_offset0 = p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;

#if __AVX2__
                    __m256i _vindex0 = _mm256_add_epi32(_dxy_offset, _mm256_set1_epi32(puv_offset0));

                    __m128i _p0 = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)bottom_blob, _vindex0, sizeof(signed char)));

                    _mm_storel_epi64((__m128i*)pp, _p0);

#else // __AVX2__

                    __m128i _vindex0 = _mm_add_epi32(_dxy_offset0, _mm_set1_epi32(puv_offset0));
                    __m128i _vindex1 = _mm_add_epi32(_dxy_offset1, _mm_set1_epi32(puv_offset0));

                    int offsets0[4];
                    int offsets1[4];
                    _mm_storeu_si128((__m128i*)offsets0, _vindex0);
                    _mm_storeu_si128((__m128i*)offsets1, _vindex1);

                    pp[0] = ((const signed char*)bottom_blob)[offsets0[0]];
                    pp[1] = ((const signed char*)bottom_blob)[offsets0[1]];
                    pp[2] = ((const signed char*)bottom_blob)[offsets0[2]];
                    pp[3] = ((const signed char*)bottom_blob)[offsets0[3]];
                    pp[4] = ((const signed char*)bottom_blob)[offsets1[0]];
                    pp[5] = ((const signed char*)bottom_blob)[offsets1[1]];
                    pp[6] = ((const signed char*)bottom_blob)[offsets1[2]];
                    pp[7] = ((const signed char*)bottom_blob)[offsets1[3]];

#endif // __AVX2__

                    pp += 8;
                }
            }
            if (elempack == 8)
            {
                int kk = 0;
                for (; kk < max_kk / 8; kk++)
                {
                    int p = (k / 8 + kk) / maxk;
                    int uv = (k / 8 + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int puv_offset = p * cstep + u * dilation_h * w + v * dilation_w;

#if __AVX2__
                    __m256i _vindex = _mm256_add_epi32(_dxy_offset, _mm256_set1_epi32(puv_offset));

                    _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(8));

                    __m256i _r0 = _mm256_i32gather_epi64((const long long int*)bottom_blob, _mm256_extractf128_si256(_vindex, 0), sizeof(signed char));
                    __m256i _r1 = _mm256_i32gather_epi64((const long long int*)bottom_blob, _mm256_extractf128_si256(_vindex, 1), sizeof(signed char));

#if __AVX512VNNI__ || __AVXVNNI__

                    // 0011 2233
                    // 4455 6677

                    __m256i _r2 = _mm256_unpacklo_epi32(_r0, _r1);
                    __m256i _r3 = _mm256_unpackhi_epi32(_r0, _r1);

                    // 0404 2626
                    // 1515 3737

                    _r0 = _mm256_unpacklo_epi32(_r2, _r3);
                    _r1 = _mm256_unpackhi_epi32(_r2, _r3);

                    // 0145 2367
                    // 0145 2367

                    _r0 = _mm256_permute4x64_epi64(_r0, _MM_SHUFFLE(3, 1, 2, 0));
                    _r1 = _mm256_permute4x64_epi64(_r1, _MM_SHUFFLE(3, 1, 2, 0));

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    __m256i _v127 = _mm256_set1_epi8(127);
                    _r0 = _mm256_add_epi8(_r0, _v127);
                    _r1 = _mm256_add_epi8(_r1, _v127);
#endif // __AVXVNNIINT8__

#else // __AVX512VNNI__ || __AVXVNNI__

                    // 00001111 22223333
                    // 44445555 66667777

                    __m256i _r2 = _mm256_unpacklo_epi16(_r0, _r1);
                    __m256i _r3 = _mm256_unpackhi_epi16(_r0, _r1);

                    // 04040404 26262626
                    // 15151515 37373737

                    _r0 = _mm256_unpacklo_epi16(_r2, _r3);
                    _r1 = _mm256_unpackhi_epi16(_r2, _r3);

                    // 01450145 23672367
                    // 01450145 23672367

                    _r0 = _mm256_permute4x64_epi64(_r0, _MM_SHUFFLE(3, 1, 2, 0));
                    _r1 = _mm256_permute4x64_epi64(_r1, _MM_SHUFFLE(3, 1, 2, 0));

                    // 01452367 01452367
                    // 01452367 01452367

                    _r0 = _mm256_shuffle_epi32(_r0, _MM_SHUFFLE(3, 1, 2, 0));
                    _r1 = _mm256_shuffle_epi32(_r1, _MM_SHUFFLE(3, 1, 2, 0));

#endif // __AVX512VNNI__ || __AVXVNNI__

                    _mm256_storeu_si256((__m256i*)pp, _r0);
                    _mm256_storeu_si256((__m256i*)(pp + 32), _r1);
#else  // __AVX2__

                    __m128i _vindex0 = _mm_add_epi32(_dxy_offset0, _mm_set1_epi32(puv_offset));
                    __m128i _vindex1 = _mm_add_epi32(_dxy_offset1, _mm_set1_epi32(puv_offset));

                    _vindex0 = _mm_comp_mullo_epi32(_vindex0, _mm_set1_epi32(8));
                    _vindex1 = _mm_comp_mullo_epi32(_vindex1, _mm_set1_epi32(8));

                    int offsets0[4];
                    int offsets1[4];
                    _mm_storeu_si128((__m128i*)offsets0, _vindex0);
                    _mm_storeu_si128((__m128i*)offsets1, _vindex1);

                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets0[0]));
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets0[1]));
                    __m128i _r2 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets0[2]));
                    __m128i _r3 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets0[3]));
                    __m128i _r4 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets1[0]));
                    __m128i _r5 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets1[1]));
                    __m128i _r6 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets1[2]));
                    __m128i _r7 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets1[3]));

                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                    __m128i _r23 = _mm_unpacklo_epi16(_r2, _r3);
                    __m128i _r45 = _mm_unpacklo_epi16(_r4, _r5);
                    __m128i _r67 = _mm_unpacklo_epi16(_r6, _r7);
                    _r0 = _mm_unpacklo_epi32(_r01, _r23);
                    _r1 = _mm_unpackhi_epi32(_r01, _r23);
                    _r2 = _mm_unpacklo_epi32(_r45, _r67);
                    _r3 = _mm_unpackhi_epi32(_r45, _r67);
                    _r4 = _mm_unpacklo_epi64(_r0, _r2);
                    _r5 = _mm_unpackhi_epi64(_r0, _r2);
                    _r6 = _mm_unpacklo_epi64(_r1, _r3);
                    _r7 = _mm_unpackhi_epi64(_r1, _r3);

                    _mm_storeu_si128((__m128i*)pp, _r4);
                    _mm_storeu_si128((__m128i*)(pp + 16), _r5);
                    _mm_storeu_si128((__m128i*)(pp + 32), _r6);
                    _mm_storeu_si128((__m128i*)(pp + 48), _r7);
#endif // __AVX2__
                    pp += 64;
                }
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        __m128i _dy;
        __m128i _dx;
        {
            __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
            __m128i _dxy = _mm_add_epi32(_mm_set1_epi32(j + jj), _offset);
            _dy = div_outw._mm_comp_div_epu32(_dxy);
            _dx = _mm_sub_epi32(_dxy, _mm_comp_mullo_epi32(_dy, _mm_set1_epi32(outw)));
            _dy = _mm_comp_mullo_epi32(_dy, _mm_set1_epi32(stride_h));
            _dx = _mm_comp_mullo_epi32(_dx, _mm_set1_epi32(stride_w));
            _dy = _mm_comp_mullo_epi32(_dy, _mm_set1_epi32(w));
        }

        __m128i _dxy_offset = _mm_add_epi32(_dx, _dy);

        const int dy0 = _mm_cvtsi128_si32(_dy);
        const int dy3 = _mm_cvtsi128_si32(_mm_shuffle_epi32(_dy, _MM_SHUFFLE(3, 3, 3, 3)));

        if (dy0 == dy3 && stride_w == 1)
        {
            const int dx0 = _mm_cvtsi128_si32(_dx);

            const int dxy_offset = dx0 + dy0;

            if (elempack == 1)
            {
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _puv_offset;
                    {
                        __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                        __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                        __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                        __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                        __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                        _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                        _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                        _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                    }

                    __m128i _offset = _mm_add_epi32(_mm_set1_epi32(dxy_offset), _puv_offset);

                    __m128i _r0 = _mm_i32gather_epi32((const int*)bottom_blob, _offset, sizeof(signed char));

                    // 0000111122223333

                    _r0 = _mm_shuffle_epi8(_r0, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));

#if __AVXVNNIINT8__
#else
                    __m128i _v127 = _mm_set1_epi8(127);
                    _r0 = _mm_add_epi8(_r0, _v127);
#endif

                    _mm_storeu_si128((__m128i*)pp, _r0);
                    pp += 16;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                    int offset1 = dxy_offset + p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offset0));
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offset1));
                    __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                    _mm_storel_epi64((__m128i*)pp, _r01);
                    pp += 8;
                }
                for (; kk < max_kk; kk += 1)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;

                    int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;

                    pp[0] = ((const signed char*)bottom_blob)[offset0];
                    pp[1] = ((const signed char*)bottom_blob)[offset0 + 1];
                    pp[2] = ((const signed char*)bottom_blob)[offset0 + 2];
                    pp[3] = ((const signed char*)bottom_blob)[offset0 + 3];

                    pp += 4;
                }
            }
            if (elempack == 8)
            {
                int kk = 0;
                for (; kk < max_kk / 8; kk++)
                {
                    int p = (k / 8 + kk) / maxk;
                    int uv = (k / 8 + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int offset = (dxy_offset + p * cstep + u * dilation_h * w + v * dilation_w) * 8;

                    __m128i _r0 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset));
                    __m128i _r1 = _mm_loadu_si128((const __m128i*)((const signed char*)bottom_blob + offset + 16));

#if __AVX512VNNI__ || __AVXVNNI__

                    // 0011
                    // 2233

                    __m128i _r2 = _mm_unpacklo_epi32(_r0, _r1);
                    __m128i _r3 = _mm_unpackhi_epi32(_r0, _r1);

                    // 0202
                    // 1313

                    _r0 = _mm_unpacklo_epi32(_r2, _r3);
                    _r1 = _mm_unpackhi_epi32(_r2, _r3);

                    // 0123
                    // 0123

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    __m128i _v127 = _mm_set1_epi8(127);
                    _r0 = _mm_add_epi8(_r0, _v127);
                    _r1 = _mm_add_epi8(_r1, _v127);
#endif // __AVXVNNIINT8__

#else // __AVX512VNNI__ || __AVXVNNI__

                    // 00001111
                    // 22223333

                    __m128i _r2 = _mm_unpacklo_epi16(_r0, _r1);
                    __m128i _r3 = _mm_unpackhi_epi16(_r0, _r1);

                    // 02020202
                    // 13131313

                    _r0 = _mm_unpacklo_epi16(_r2, _r3);
                    _r1 = _mm_unpackhi_epi16(_r2, _r3);

#endif // __AVX512VNNI__ || __AVXVNNI__

                    _mm_storeu_si128((__m128i*)pp, _r0);
                    _mm_storeu_si128((__m128i*)(pp + 16), _r1);
                    pp += 32;
                }
            }
        }
        else
        {
            if (elempack == 1)
            {
                int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _puv_offset;
                    {
                        __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                        __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                        __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                        __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                        __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                        _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                        _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                        _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                    }

                    __m128i _vindex0 = _mm_add_epi32(_dxy_offset, _mm_set1_epi32(_mm_extract_epi32(_puv_offset, 0)));
                    __m128i _vindex1 = _mm_add_epi32(_dxy_offset, _mm_set1_epi32(_mm_extract_epi32(_puv_offset, 1)));
                    __m128i _vindex2 = _mm_add_epi32(_dxy_offset, _mm_set1_epi32(_mm_extract_epi32(_puv_offset, 2)));
                    __m128i _vindex3 = _mm_add_epi32(_dxy_offset, _mm_set1_epi32(_mm_extract_epi32(_puv_offset, 3)));

                    __m128i _p0 = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)bottom_blob, _vindex0, sizeof(signed char)));
                    __m128i _p1 = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)bottom_blob, _vindex1, sizeof(signed char)));
                    __m128i _p2 = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)bottom_blob, _vindex2, sizeof(signed char)));
                    __m128i _p3 = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)bottom_blob, _vindex3, sizeof(signed char)));

                    // 0000............
                    // 1111............
                    // 2222............
                    // 3333............

                    __m128i _p01 = _mm_unpacklo_epi8(_p0, _p1);
                    __m128i _p23 = _mm_unpacklo_epi8(_p2, _p3);

                    // 01010101 ........
                    // 23232323 ........

                    __m128i _p0123 = _mm_unpacklo_epi16(_p01, _p23);

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    __m128i _v127 = _mm_set1_epi8(127);
                    _p0123 = _mm_add_epi8(_p0123, _v127);
#endif // __AVXVNNIINT8__

                    _mm_storeu_si128((__m128i*)pp, _p0123);

                    pp += 16;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    int puv_offset0 = p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                    int puv_offset1 = p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

                    __m128i _vindex0 = _mm_add_epi32(_dxy_offset, _mm_set1_epi32(puv_offset0));
                    __m128i _vindex1 = _mm_add_epi32(_dxy_offset, _mm_set1_epi32(puv_offset1));

#if __AVX2__
                    __m128i _p0 = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)bottom_blob, _vindex0, sizeof(signed char)));
                    __m128i _p1 = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)bottom_blob, _vindex1, sizeof(signed char)));

                    __m128i _p01 = _mm_unpacklo_epi8(_p0, _p1);

                    _mm_storel_epi64((__m128i*)pp, _p01);
#else
                    int offsets0[4];
                    int offsets1[4];
                    _mm_storeu_si128((__m128i*)offsets0, _vindex0);
                    _mm_storeu_si128((__m128i*)offsets1, _vindex1);

                    pp[0] = ((const signed char*)bottom_blob)[offsets0[0]];
                    pp[1] = ((const signed char*)bottom_blob)[offsets1[0]];
                    pp[2] = ((const signed char*)bottom_blob)[offsets0[1]];
                    pp[3] = ((const signed char*)bottom_blob)[offsets1[1]];
                    pp[4] = ((const signed char*)bottom_blob)[offsets0[2]];
                    pp[5] = ((const signed char*)bottom_blob)[offsets1[2]];
                    pp[6] = ((const signed char*)bottom_blob)[offsets0[3]];
                    pp[7] = ((const signed char*)bottom_blob)[offsets1[3]];
#endif

                    pp += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    int p = (k + kk) / maxk;
                    int uv = (k + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int puv_offset = p * cstep + u * dilation_h * w + v * dilation_w;

                    __m128i _vindex0 = _mm_add_epi32(_dxy_offset, _mm_set1_epi32(puv_offset));

#if __AVX2__
                    __m128i _p0 = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)bottom_blob, _vindex0, sizeof(signed char)));

                    _mm_store_ss((float*)pp, _mm_castsi128_ps(_p0));
#else
                    int offsets[4];
                    _mm_storeu_si128((__m128i*)offsets, _vindex0);

                    pp[0] = ((const signed char*)bottom_blob)[offsets[0]];
                    pp[1] = ((const signed char*)bottom_blob)[offsets[1]];
                    pp[2] = ((const signed char*)bottom_blob)[offsets[2]];
                    pp[3] = ((const signed char*)bottom_blob)[offsets[3]];
#endif

                    pp += 4;
                }
            }
            if (elempack == 8)
            {
                int kk = 0;
                for (; kk < max_kk / 8; kk++)
                {
                    int p = (k / 8 + kk) / maxk;
                    int uv = (k / 8 + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int puv_offset = p * cstep + u * dilation_h * w + v * dilation_w;

                    __m128i _vindex = _mm_add_epi32(_dxy_offset, _mm_set1_epi32(puv_offset));

                    _vindex = _mm_comp_mullo_epi32(_vindex, _mm_set1_epi32(8));

#if __AVX2__
                    __m256i _r01 = _mm256_i32gather_epi64((const long long int*)bottom_blob, _vindex, sizeof(signed char));

#if __AVX512VNNI__ || __AVXVNNI__
                    // 0011 2233
                    _r01 = _mm256_shuffle_epi32(_r01, _MM_SHUFFLE(3, 1, 2, 0));
                    _r01 = _mm256_permute4x64_epi64(_r01, _MM_SHUFFLE(3, 1, 2, 0));

#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    __m256i _v127 = _mm256_set1_epi8(127);
                    _r01 = _mm256_add_epi8(_r01, _v127);
#endif // __AVXVNNIINT8__

#else // __AVX512VNNI__ || __AVXVNNI__

                    // 00001111 22223333
                    _r01 = _mm256_shuffle_epi32(_r01, _MM_SHUFFLE(3, 1, 2, 0));

                    // 00110011 22332233
                    _r01 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_r01, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));

                    // 01010101 23232323
                    _r01 = _mm256_permute4x64_epi64(_r01, _MM_SHUFFLE(3, 1, 2, 0));

                    // 01012323 01012323
                    _r01 = _mm256_shuffle_epi32(_r01, _MM_SHUFFLE(3, 1, 2, 0));

#endif // __AVX512VNNI__ || __AVXVNNI__

                    _mm256_storeu_si256((__m256i*)pp, _r01);

#else  // __AVX2__

                    int offsets[4];
                    _mm_storeu_si128((__m128i*)offsets, _vindex);

                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets[0]));
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets[1]));
                    __m128i _r2 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets[2]));
                    __m128i _r3 = _mm_loadl_epi64((const __m128i*)((const signed char*)bottom_blob + offsets[3]));

                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                    __m128i _r23 = _mm_unpacklo_epi16(_r2, _r3);
                    _r0 = _mm_unpacklo_epi32(_r01, _r23);
                    _r1 = _mm_unpackhi_epi32(_r01, _r23);

                    _mm_storeu_si128((__m128i*)pp, _r0);
                    _mm_storeu_si128((__m128i*)(pp + 16), _r1);
#endif // __AVX2__

                    pp += 32;
                }
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        dy0 *= stride_h;
        dy1 *= stride_h;
        dx0 *= stride_w;
        dx1 *= stride_w;

        dy0 *= w;
        dy1 *= w;

        const int dxy_offset0 = dx0 + dy0;
        const int dxy_offset1 = dx1 + dy1;

        if (dy0 == dy1 && stride_w == 1)
        {
            const int dxy_offset = dxy_offset0;

            if (elempack == 1)
            {
                int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _puv_offset;
                    {
                        __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                        __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                        __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                        __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                        __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                        _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                        _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                        _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                    }

                    __m128i _offset = _mm_add_epi32(_mm_set1_epi32(dxy_offset), _puv_offset);

                    __m128i _r0 = _mm_comp_cvtepi32_epi16(_mm_i32gather_epi32((const int*)bottom_blob, _offset, sizeof(signed char)));

                    // 00112233........

                    _r0 = _mm_shuffle_epi8(_r0, _mm_setr_epi8(0, 2, 4, 6, 1, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0));

#if __AVXVNNIINT8__
#else // __AVXVNNIINT8__

                    __m128i _v127 = _mm_set1_epi8(127);

                    _r0 = _mm_add_epi8(_r0, _v127);

#endif // __AVXVNNIINT8__

                    _mm_storel_epi64((__m128i*)pp, _r0);

                    pp += 8;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                    int offset1 = dxy_offset + p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

                    const signed char* sptr0 = (const signed char*)bottom_blob + offset0;
                    const signed char* sptr1 = (const signed char*)bottom_blob + offset1;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[1];
                    pp[3] = sptr1[1];
                    pp += 4;
                }
#endif // __SSE2__
                for (; kk < max_kk; kk += 1)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;

                    int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;

                    const signed char* sptr0 = (const signed char*)bottom_blob + offset0;

                    pp[0] = sptr0[0];
                    pp[1] = sptr0[1];
                    pp += 2;
                }
            }
#if __SSE2__
            if (elempack == 8)
            {
                int kk = 0;
                for (; kk < max_kk / 8; kk++)
                {
                    int p = (k / 8 + kk) / maxk;
                    int uv = (k / 8 + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int offset = (dxy_offset + p * cstep + u * dilation_h * w + v * dilation_w) * 8;

                    const signed char* sptr = (const signed char*)bottom_blob + offset;

                    __m128i _r0 = _mm_loadu_si128((const __m128i*)sptr);

#if __AVX512VNNI__ || __AVXVNNI__
                    _r0 = _mm_shuffle_epi32(_r0, _MM_SHUFFLE(3, 1, 2, 0));
#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    _r0 = _mm_add_epi8(_r0, _mm_set1_epi8(127));
#endif // __AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__

                    // 00001111
                    _r0 = _mm_shuffle_epi32(_r0, _MM_SHUFFLE(3, 1, 2, 0));

                    // 00110011
                    _r0 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_r0, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));

#endif // __AVX512VNNI__ || __AVXVNNI__
                    _mm_storeu_si128((__m128i*)pp, _r0);
                    pp += 16;
                }
            }
#endif // __SSE2__
        }
        else
        {
            int kk = 0;
            if (elempack == 1)
            {
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _puv_offset;
                    {
                        __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                        __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                        __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                        __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                        __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                        _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                        _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                        _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                        _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                    }

                    int puv_offset[4];
                    _mm_storeu_si128((__m128i*)puv_offset, _puv_offset);

                    int offset00 = dxy_offset0 + puv_offset[0];
                    int offset01 = dxy_offset1 + puv_offset[0];
                    int offset10 = dxy_offset0 + puv_offset[1];
                    int offset11 = dxy_offset1 + puv_offset[1];
                    int offset20 = dxy_offset0 + puv_offset[2];
                    int offset21 = dxy_offset1 + puv_offset[2];
                    int offset30 = dxy_offset0 + puv_offset[3];
                    int offset31 = dxy_offset1 + puv_offset[3];

                    const signed char* sptr00 = (const signed char*)bottom_blob + offset00;
                    const signed char* sptr01 = (const signed char*)bottom_blob + offset01;
                    const signed char* sptr10 = (const signed char*)bottom_blob + offset10;
                    const signed char* sptr11 = (const signed char*)bottom_blob + offset11;
                    const signed char* sptr20 = (const signed char*)bottom_blob + offset20;
                    const signed char* sptr21 = (const signed char*)bottom_blob + offset21;
                    const signed char* sptr30 = (const signed char*)bottom_blob + offset30;
                    const signed char* sptr31 = (const signed char*)bottom_blob + offset31;

#if __AVXVNNIINT8__
                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr01[0];
                    pp[5] = sptr11[0];
                    pp[6] = sptr21[0];
                    pp[7] = sptr31[0];
#else  // __AVXVNNIINT8__
                    pp[0] = sptr00[0] + 127;
                    pp[1] = sptr10[0] + 127;
                    pp[2] = sptr20[0] + 127;
                    pp[3] = sptr30[0] + 127;
                    pp[4] = sptr01[0] + 127;
                    pp[5] = sptr11[0] + 127;
                    pp[6] = sptr21[0] + 127;
                    pp[7] = sptr31[0] + 127;
#endif // __AVXVNNIINT8__
                    pp += 8;
                }
#endif // __AVX512VNNI__ || __AVXVNNI__
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    int puv_offset0 = p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                    int puv_offset1 = p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

                    int offset00 = dxy_offset0 + puv_offset0;
                    int offset01 = dxy_offset1 + puv_offset0;
                    int offset10 = dxy_offset0 + puv_offset1;
                    int offset11 = dxy_offset1 + puv_offset1;

                    const signed char* sptr00 = (const signed char*)bottom_blob + offset00;
                    const signed char* sptr01 = (const signed char*)bottom_blob + offset01;
                    const signed char* sptr10 = (const signed char*)bottom_blob + offset10;
                    const signed char* sptr11 = (const signed char*)bottom_blob + offset11;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp += 4;
                }
#endif // __SSE2__
                for (; kk < max_kk; kk += 1)
                {
                    int p = (k + kk) / maxk;
                    int uv = (k + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int offset0 = dxy_offset0 + p * cstep + u * dilation_h * w + v * dilation_w;
                    int offset1 = dxy_offset1 + p * cstep + u * dilation_h * w + v * dilation_w;

                    const signed char* sptr00 = (const signed char*)bottom_blob + offset0;
                    const signed char* sptr01 = (const signed char*)bottom_blob + offset1;

                    pp[0] = sptr00[0];
                    pp[1] = sptr01[0];
                    pp += 2;
                }
            }
#if __SSE2__
            if (elempack == 8)
            {
                for (; kk < max_kk / 8; kk++)
                {
                    int p = (k / 8 + kk) / maxk;
                    int uv = (k / 8 + kk) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    int offset0 = (dxy_offset0 + p * cstep + u * dilation_h * w + v * dilation_w) * 8;
                    int offset1 = (dxy_offset1 + p * cstep + u * dilation_h * w + v * dilation_w) * 8;

                    const signed char* sptr0 = (const signed char*)bottom_blob + offset0;
                    const signed char* sptr1 = (const signed char*)bottom_blob + offset1;

                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
#if __AVX512VNNI__ || __AVXVNNI__
                    __m128i _r01 = _mm_unpacklo_epi32(_r0, _r1);
#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                    _r01 = _mm_add_epi8(_r01, _mm_set1_epi8(127));
#endif // __AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
#endif // __AVX512VNNI__ || __AVXVNNI__
                    _mm_storeu_si128((__m128i*)pp, _r01);
                    pp += 16;
                }
            }
#endif // __SSE2__
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        dy *= stride_h;
        dx *= stride_w;

        dy *= w;

        const int dxy_offset = dx + dy;

        if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _puv_offset;
                {
                    __m128i _offset = _mm_setr_epi32(0, 1, 2, 3);
                    __m128i _puv = _mm_add_epi32(_mm_set1_epi32(k + kk), _offset);
                    __m128i _p = div_maxk._mm_comp_div_epu32(_puv);
                    __m128i _uv = _mm_sub_epi32(_puv, _mm_mullo_epi32(_p, _mm_set1_epi32(maxk)));
                    __m128i _u = div_kernel_w._mm_comp_div_epu32(_uv);
                    __m128i _v = _mm_sub_epi32(_uv, _mm_mullo_epi32(_u, _mm_set1_epi32(kernel_w)));
                    _p = _mm_mullo_epi32(_p, _mm_set1_epi32(cstep));
                    _u = _mm_mullo_epi32(_u, _mm_set1_epi32(dilation_h));
                    _v = _mm_mullo_epi32(_v, _mm_set1_epi32(dilation_w));
                    _u = _mm_mullo_epi32(_u, _mm_set1_epi32(w));
                    _puv_offset = _mm_add_epi32(_p, _mm_add_epi32(_u, _v));
                }

                int puv_offset[4];
                _mm_storeu_si128((__m128i*)puv_offset, _puv_offset);

                int offset0 = dxy_offset + puv_offset[0];
                int offset1 = dxy_offset + puv_offset[1];
                int offset2 = dxy_offset + puv_offset[2];
                int offset3 = dxy_offset + puv_offset[3];

                const signed char* sptr0 = (const signed char*)bottom_blob + offset0;
                const signed char* sptr1 = (const signed char*)bottom_blob + offset1;
                const signed char* sptr2 = (const signed char*)bottom_blob + offset2;
                const signed char* sptr3 = (const signed char*)bottom_blob + offset3;

#if __AVXVNNIINT8__
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr2[0];
                pp[3] = sptr3[0];
#else  // __AVXVNNIINT8__
                pp[0] = sptr0[0] + 127;
                pp[1] = sptr1[0] + 127;
                pp[2] = sptr2[0] + 127;
                pp[3] = sptr3[0] + 127;
#endif // __AVXVNNIINT8__
                pp += 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;

                int offset0 = dxy_offset + p0 * cstep + u0 * dilation_h * w + v0 * dilation_w;
                int offset1 = dxy_offset + p1 * cstep + u1 * dilation_h * w + v1 * dilation_w;

                const signed char* sptr0 = (const signed char*)bottom_blob + offset0;
                const signed char* sptr1 = (const signed char*)bottom_blob + offset1;

                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp += 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk += 1)
            {
                int p = (k + kk) / maxk;
                int uv = (k + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                int offset = dxy_offset + p * cstep + u * dilation_h * w + v * dilation_w;

                const signed char* sptr0 = (const signed char*)bottom_blob + offset;

                pp[0] = sptr0[0];
                pp += 1;
            }
        }
#if __SSE2__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                int p = (k / 8 + kk) / maxk;
                int uv = (k / 8 + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                int offset = (dxy_offset + p * cstep + u * dilation_h * w + v * dilation_w) * 8;

                const signed char* sptr = (const signed char*)bottom_blob + offset;

                __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
#else  // __AVXVNNIINT8__
                _r0 = _mm_add_epi8(_r0, _mm_set1_epi8(127));
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
                _mm_storel_epi64((__m128i*)pp, _r0);
                pp += 8;
            }
        }
#endif // __SSE2__
    }
}

static void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        convolution_im2col_input_tile_int8_avx512vnni(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        convolution_im2col_input_tile_int8_avxvnniint8(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        convolution_im2col_input_tile_int8_avxvnni(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        convolution_im2col_input_tile_int8_avx2(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
        return;
    }
#endif

    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_int8(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    // TODO specialized template initialization for 1x1s2 3x3s1 3x2s2 5x5s1 5x5s2 7x7s2

    convolution_im2col_input_tile_int8_impl(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel_int8");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        elempack = inch % 8 == 0 ? 8 : 1;
    }
#endif // __SSE2__

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        A_data = kernel.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)1u, 1);

        for (int q = 0; q < outch; q += 1)
        {
            signed char* g00 = A_data.row<signed char>(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const signed char* k00 = weight_data_r2.channel(q).row<const signed char>(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }

#if NCNN_AVX512VNNI || NCNN_AVXVNNI
    bool has_w_shift = false;
    if (TILE_K >= 4)
    {
        has_w_shift = ncnn::cpu_support_x86_avx512_vnni() || ncnn::cpu_support_x86_avx_vnni();
#if NCNN_AVXVNNIINT8
        if (ncnn::cpu_support_x86_avx_vnni_int8())
            has_w_shift = false;
#endif // NCNN_AVXVNNIINT8
    }
    if (has_w_shift)
    {
        int w_shift_count = TILE_M >= 16 ? 16 : TILE_M >= 8 ? 8 : TILE_M >= 4 ? 4 : TILE_M >= 2 ? 2 : 1;
        AT.create((TILE_K + w_shift_count * 4) * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)1u, 1);
    }
    else
#endif // NCNN_AVX512VNNI || NCNN_AVXVNNI
    {
        AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)1u, 1);
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static void unpack_output_tile_int32(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        unpack_output_tile_int32_avx2(topT, top_blob, i, max_ii, j, max_jj);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    // const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;
    const int out_hstep = top_blob.cstep;

    // NCNN_LOGE("unpack_output_tile_int32_to_fp32  %d %d %d %d   @%d", i, max_ii, j, max_jj, out_elempack);

    const int* pp = topT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _f0 = _mm512_load_si512((const __m512i*)pp);
            __m512i _f1 = _mm512_load_si512((const __m512i*)(pp + 16));
            __m512i _f2 = _mm512_load_si512((const __m512i*)(pp + 32));
            __m512i _f3 = _mm512_load_si512((const __m512i*)(pp + 48));
            __m512i _f4 = _mm512_load_si512((const __m512i*)(pp + 64));
            __m512i _f5 = _mm512_load_si512((const __m512i*)(pp + 80));
            __m512i _f6 = _mm512_load_si512((const __m512i*)(pp + 96));
            __m512i _f7 = _mm512_load_si512((const __m512i*)(pp + 112));
            __m512i _f8 = _mm512_load_si512((const __m512i*)(pp + 128));
            __m512i _f9 = _mm512_load_si512((const __m512i*)(pp + 128 + 16));
            __m512i _fa = _mm512_load_si512((const __m512i*)(pp + 128 + 32));
            __m512i _fb = _mm512_load_si512((const __m512i*)(pp + 128 + 48));
            __m512i _fc = _mm512_load_si512((const __m512i*)(pp + 128 + 64));
            __m512i _fd = _mm512_load_si512((const __m512i*)(pp + 128 + 80));
            __m512i _fe = _mm512_load_si512((const __m512i*)(pp + 128 + 96));
            __m512i _ff = _mm512_load_si512((const __m512i*)(pp + 128 + 112));
            pp += 256;

            // from
            // 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
            // 01 12 23 30 45 56 67 74 89 9a ab b8 cd de ef fc
            // 20 31 02 13 64 75 46 57 a8 b9 8a 9b ec fd ce df
            // 21 32 03 10 65 76 47 54 a9 ba 8b 98 ed fe cf dc
            // 08 19 2a 3b 4c 5d 6e 7f 80 91 a2 b3 c4 d5 e6 f7
            // 09 1a 2b 38 4d 5e 6f 7c 81 92 a3 b0 c5 d6 e7 f4
            // 28 39 0a 1b 6c 7d 4e 5f a0 b1 82 93 e4 f5 c6 d7
            // 29 3a 0b 18 6d 7e 4f 5c a1 b2 83 90 e5 f6 c7 d4
            // 40 51 62 73 04 15 26 37 c8 d9 ea fb 8c 9d ae bf
            // 41 52 63 70 05 16 27 34 c9 da eb f8 8d 9e af bc
            // 60 71 42 53 24 35 06 17 e8 f9 ca db ac bd 8e 9f
            // 61 72 43 50 25 36 07 14 e9 fa cb d8 ad be 8f 9c
            // 48 59 6a 7b 0c 1d 2e 3f c0 d1 e2 f3 84 95 a6 b7
            // 49 5a 6b 78 0d 1e 2f 3c c1 d2 e3 f0 85 96 a7 b4
            // 68 79 4a 5b 2c 3d 0e 1f e0 f1 c2 d3 a4 b5 86 97
            // 69 7a 4b 58 2d 3e 0f 1c e1 f2 c3 d0 a5 b6 87 94

            // to
            // 00 10 20 30  40 50 60 70  80 90 a0 b0  c0 d0 e0 f0
            // 01 11 21 31  41 51 61 71  81 91 a1 b1  c1 d1 e1 f1
            // 02 12 22 32  42 52 62 72  82 92 a2 b2  c2 d2 e2 f2
            // 03 13 23 33  43 53 63 73  83 93 a3 b3  c3 d3 e3 f3
            // 04 14 24 34  44 54 64 74  84 94 a4 b4  c4 d4 e4 f4
            // 05 15 25 35  45 55 65 75  85 95 a5 b5  c5 d5 e5 f5
            // 06 16 26 36  46 56 66 76  86 96 a6 b6  c6 d6 e6 f6
            // 07 17 27 37  47 57 67 77  87 97 a7 b7  c7 d7 e7 f7
            // 08 18 28 38  48 58 68 78  88 98 a8 b8  c8 d8 e8 f8
            // 09 19 29 39  49 59 69 79  89 99 a9 b9  c9 d9 e9 f9
            // 0a 1a 2a 3a  4a 5a 6a 7a  8a 9a aa ba  ca da ea fa
            // 0b 1b 2b 3b  4b 5b 6b 7b  8b 9b ab bb  cb db eb fb
            // 0c 1c 2c 3c  4c 5c 6c 7c  8c 9c ac bc  cc dc ec fc
            // 0d 1d 2d 3d  4d 5d 6d 7d  8d 9d ad bd  cd dd ed fd
            // 0e 1e 2e 3e  4e 5e 6e 7e  8e 9e ae be  ce de ee fe
            // 0f 1f 2f 3f  4f 5f 6f 7f  8f 9f af bf  cf df ef ff
            {
                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
                _f5 = _mm512_shuffle_epi32(_f5, _MM_PERM_CBAD);
                _f7 = _mm512_shuffle_epi32(_f7, _MM_PERM_CBAD);
                _f9 = _mm512_shuffle_epi32(_f9, _MM_PERM_CBAD);
                _fb = _mm512_shuffle_epi32(_fb, _MM_PERM_CBAD);
                _fd = _mm512_shuffle_epi32(_fd, _MM_PERM_CBAD);
                _ff = _mm512_shuffle_epi32(_ff, _MM_PERM_CBAD);

                __m512i _tmp0 = _mm512_unpacklo_epi32(_f0, _f3);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_f0, _f3);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_f2, _f1);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_f2, _f1);
                __m512i _tmp4 = _mm512_unpacklo_epi32(_f4, _f7);
                __m512i _tmp5 = _mm512_unpackhi_epi32(_f4, _f7);
                __m512i _tmp6 = _mm512_unpacklo_epi32(_f6, _f5);
                __m512i _tmp7 = _mm512_unpackhi_epi32(_f6, _f5);
                __m512i _tmp8 = _mm512_unpacklo_epi32(_f8, _fb);
                __m512i _tmp9 = _mm512_unpackhi_epi32(_f8, _fb);
                __m512i _tmpa = _mm512_unpacklo_epi32(_fa, _f9);
                __m512i _tmpb = _mm512_unpackhi_epi32(_fa, _f9);
                __m512i _tmpc = _mm512_unpacklo_epi32(_fc, _ff);
                __m512i _tmpd = _mm512_unpackhi_epi32(_fc, _ff);
                __m512i _tmpe = _mm512_unpacklo_epi32(_fe, _fd);
                __m512i _tmpf = _mm512_unpackhi_epi32(_fe, _fd);

                _f0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _f1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _f2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                _f3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                _f4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                _f5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                _f6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                _f7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                _f8 = _mm512_unpacklo_epi64(_tmp8, _tmpa);
                _f9 = _mm512_unpackhi_epi64(_tmp8, _tmpa);
                _fa = _mm512_unpacklo_epi64(_tmpb, _tmp9);
                _fb = _mm512_unpackhi_epi64(_tmpb, _tmp9);
                _fc = _mm512_unpacklo_epi64(_tmpc, _tmpe);
                _fd = _mm512_unpackhi_epi64(_tmpc, _tmpe);
                _fe = _mm512_unpacklo_epi64(_tmpf, _tmpd);
                _ff = _mm512_unpackhi_epi64(_tmpf, _tmpd);

                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
                _f5 = _mm512_shuffle_epi32(_f5, _MM_PERM_CBAD);
                _f7 = _mm512_shuffle_epi32(_f7, _MM_PERM_CBAD);
                _f9 = _mm512_shuffle_epi32(_f9, _MM_PERM_CBAD);
                _fb = _mm512_shuffle_epi32(_fb, _MM_PERM_CBAD);
                _fd = _mm512_shuffle_epi32(_fd, _MM_PERM_CBAD);
                _ff = _mm512_shuffle_epi32(_ff, _MM_PERM_CBAD);

                _tmp0 = _mm512_shuffle_i32x4(_f0, _f8, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp1 = _mm512_shuffle_i32x4(_f1, _f9, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp2 = _mm512_shuffle_i32x4(_f2, _fa, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp3 = _mm512_shuffle_i32x4(_f3, _fb, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp4 = _mm512_shuffle_i32x4(_f8, _f0, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp5 = _mm512_shuffle_i32x4(_f9, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp6 = _mm512_shuffle_i32x4(_fa, _f2, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp7 = _mm512_shuffle_i32x4(_fb, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp8 = _mm512_shuffle_i32x4(_f4, _fc, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp9 = _mm512_shuffle_i32x4(_f5, _fd, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpa = _mm512_shuffle_i32x4(_f6, _fe, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpb = _mm512_shuffle_i32x4(_f7, _ff, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpc = _mm512_shuffle_i32x4(_fc, _f4, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpd = _mm512_shuffle_i32x4(_fd, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpe = _mm512_shuffle_i32x4(_fe, _f6, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpf = _mm512_shuffle_i32x4(_ff, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 1, 2, 0));
                _f1 = _mm512_shuffle_i32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 1, 2, 0));
                _f2 = _mm512_shuffle_i32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 1, 2, 0));
                _f3 = _mm512_shuffle_i32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 1, 2, 0));
                _f4 = _mm512_shuffle_i32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 1, 2, 0));
                _f5 = _mm512_shuffle_i32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 1, 2, 0));
                _f6 = _mm512_shuffle_i32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 1, 2, 0));
                _f7 = _mm512_shuffle_i32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 1, 2, 0));
                _f8 = _mm512_shuffle_i32x4(_tmp8, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                _f9 = _mm512_shuffle_i32x4(_tmp9, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                _fa = _mm512_shuffle_i32x4(_tmpa, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                _fb = _mm512_shuffle_i32x4(_tmpb, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                _fc = _mm512_shuffle_i32x4(_tmpc, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                _fd = _mm512_shuffle_i32x4(_tmpd, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                _fe = _mm512_shuffle_i32x4(_tmpe, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                _ff = _mm512_shuffle_i32x4(_tmpf, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));
            }

            {
                if (out_elempack == 16)
                {
                    _mm512_store_si512((__m512i*)p0, _f0);
                    _mm512_store_si512((__m512i*)(p0 + 16), _f1);
                    _mm512_store_si512((__m512i*)(p0 + 32), _f2);
                    _mm512_store_si512((__m512i*)(p0 + 48), _f3);
                    _mm512_store_si512((__m512i*)(p0 + 64), _f4);
                    _mm512_store_si512((__m512i*)(p0 + 80), _f5);
                    _mm512_store_si512((__m512i*)(p0 + 96), _f6);
                    _mm512_store_si512((__m512i*)(p0 + 112), _f7);
                    _mm512_store_si512((__m512i*)(p0 + 128), _f8);
                    _mm512_store_si512((__m512i*)(p0 + 128 + 16), _f9);
                    _mm512_store_si512((__m512i*)(p0 + 128 + 32), _fa);
                    _mm512_store_si512((__m512i*)(p0 + 128 + 48), _fb);
                    _mm512_store_si512((__m512i*)(p0 + 128 + 64), _fc);
                    _mm512_store_si512((__m512i*)(p0 + 128 + 80), _fd);
                    _mm512_store_si512((__m512i*)(p0 + 128 + 96), _fe);
                    _mm512_store_si512((__m512i*)(p0 + 128 + 112), _ff);
                    p0 += 256;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_si256((__m256i*)p0, _mm512_extracti32x8_epi32(_f0, 0));
                    _mm256_store_si256((__m256i*)(p0 + 8), _mm512_extracti32x8_epi32(_f1, 0));
                    _mm256_store_si256((__m256i*)(p0 + 16), _mm512_extracti32x8_epi32(_f2, 0));
                    _mm256_store_si256((__m256i*)(p0 + 24), _mm512_extracti32x8_epi32(_f3, 0));
                    _mm256_store_si256((__m256i*)(p0 + 32), _mm512_extracti32x8_epi32(_f4, 0));
                    _mm256_store_si256((__m256i*)(p0 + 40), _mm512_extracti32x8_epi32(_f5, 0));
                    _mm256_store_si256((__m256i*)(p0 + 48), _mm512_extracti32x8_epi32(_f6, 0));
                    _mm256_store_si256((__m256i*)(p0 + 56), _mm512_extracti32x8_epi32(_f7, 0));
                    _mm256_store_si256((__m256i*)(p0 + 64), _mm512_extracti32x8_epi32(_f8, 0));
                    _mm256_store_si256((__m256i*)(p0 + 64 + 8), _mm512_extracti32x8_epi32(_f9, 0));
                    _mm256_store_si256((__m256i*)(p0 + 64 + 16), _mm512_extracti32x8_epi32(_fa, 0));
                    _mm256_store_si256((__m256i*)(p0 + 64 + 24), _mm512_extracti32x8_epi32(_fb, 0));
                    _mm256_store_si256((__m256i*)(p0 + 64 + 32), _mm512_extracti32x8_epi32(_fc, 0));
                    _mm256_store_si256((__m256i*)(p0 + 64 + 40), _mm512_extracti32x8_epi32(_fd, 0));
                    _mm256_store_si256((__m256i*)(p0 + 64 + 48), _mm512_extracti32x8_epi32(_fe, 0));
                    _mm256_store_si256((__m256i*)(p0 + 64 + 56), _mm512_extracti32x8_epi32(_ff, 0));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8), _mm512_extracti32x8_epi32(_f0, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 8), _mm512_extracti32x8_epi32(_f1, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 16), _mm512_extracti32x8_epi32(_f2, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 24), _mm512_extracti32x8_epi32(_f3, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 32), _mm512_extracti32x8_epi32(_f4, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 40), _mm512_extracti32x8_epi32(_f5, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 48), _mm512_extracti32x8_epi32(_f6, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 56), _mm512_extracti32x8_epi32(_f7, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 64), _mm512_extracti32x8_epi32(_f8, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 64 + 8), _mm512_extracti32x8_epi32(_f9, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 64 + 16), _mm512_extracti32x8_epi32(_fa, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 64 + 24), _mm512_extracti32x8_epi32(_fb, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 64 + 32), _mm512_extracti32x8_epi32(_fc, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 64 + 40), _mm512_extracti32x8_epi32(_fd, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 64 + 48), _mm512_extracti32x8_epi32(_fe, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 64 + 56), _mm512_extracti32x8_epi32(_ff, 1));
                    p0 += 128;
                }
                if (out_elempack == 4)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_f8, _f9, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_fa, _fb, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_fc, _fd, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_fe, _ff, _MM_SHUFFLE(2, 0, 2, 0));

                    __m512i _tmp8 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp9 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpa = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpb = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpc = _mm512_shuffle_i32x4(_f8, _f9, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpd = _mm512_shuffle_i32x4(_fa, _fb, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpe = _mm512_shuffle_i32x4(_fc, _fd, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpf = _mm512_shuffle_i32x4(_fe, _ff, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f4 = _mm512_shuffle_i32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                    _f5 = _mm512_shuffle_i32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                    _f6 = _mm512_shuffle_i32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _f7 = _mm512_shuffle_i32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));

                    _f8 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f9 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _fa = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _fb = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _fc = _mm512_shuffle_i32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                    _fd = _mm512_shuffle_i32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
                    _fe = _mm512_shuffle_i32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                    _ff = _mm512_shuffle_i32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)p0, _f0);
                    _mm512_storeu_si512((__m512i*)(p0 + 16), _f1);
                    _mm512_storeu_si512((__m512i*)(p0 + 32), _f2);
                    _mm512_storeu_si512((__m512i*)(p0 + 48), _f3);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4), _f4);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4 + 16), _f5);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4 + 32), _f6);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4 + 48), _f7);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8), _f8);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8 + 16), _f9);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8 + 32), _fa);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8 + 48), _fb);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 12), _fc);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 12 + 16), _fd);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 12 + 32), _fe);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 12 + 48), _ff);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    transpose16x16_epi32(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);

                    _mm512_storeu_si512((__m512i*)p0, _f0);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep), _f1);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 2), _f2);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 3), _f3);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4), _f4);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 5), _f5);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 6), _f6);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 7), _f7);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8), _f8);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 9), _f9);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 10), _fa);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 11), _fb);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 12), _fc);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 13), _fd);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 14), _fe);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 15), _ff);
                    p0 += 16;
                }
            }
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m512i _f0 = _mm512_load_si512((const __m512i*)pp);
            __m512i _f1 = _mm512_load_si512((const __m512i*)(pp + 16));
            __m512i _f2 = _mm512_load_si512((const __m512i*)(pp + 32));
            __m512i _f3 = _mm512_load_si512((const __m512i*)(pp + 48));
            __m512i _f4 = _mm512_load_si512((const __m512i*)(pp + 64));
            __m512i _f5 = _mm512_load_si512((const __m512i*)(pp + 80));
            __m512i _f6 = _mm512_load_si512((const __m512i*)(pp + 96));
            __m512i _f7 = _mm512_load_si512((const __m512i*)(pp + 112));
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
            {
                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
                _f5 = _mm512_shuffle_epi32(_f5, _MM_PERM_CBAD);
                _f7 = _mm512_shuffle_epi32(_f7, _MM_PERM_CBAD);

                __m512i _tmp0 = _mm512_unpacklo_epi32(_f0, _f3);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_f0, _f3);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_f2, _f1);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_f2, _f1);
                __m512i _tmp4 = _mm512_unpacklo_epi32(_f4, _f7);
                __m512i _tmp5 = _mm512_unpackhi_epi32(_f4, _f7);
                __m512i _tmp6 = _mm512_unpacklo_epi32(_f6, _f5);
                __m512i _tmp7 = _mm512_unpackhi_epi32(_f6, _f5);

                _f0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _f1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _f2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                _f3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                _f4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                _f5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                _f6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                _f7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);

                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
                _f5 = _mm512_shuffle_epi32(_f5, _MM_PERM_CBAD);
                _f7 = _mm512_shuffle_epi32(_f7, _MM_PERM_CBAD);

                _tmp0 = _mm512_shuffle_i32x4(_f0, _f4, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp1 = _mm512_shuffle_i32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp2 = _mm512_shuffle_i32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_i32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp4 = _mm512_shuffle_i32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp5 = _mm512_shuffle_i32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_i32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp7 = _mm512_shuffle_i32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));

                _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                _f1 = _mm512_shuffle_i32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _f2 = _mm512_shuffle_i32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                _f3 = _mm512_shuffle_i32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _f4 = _mm512_shuffle_i32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 3, 1, 3));
                _f5 = _mm512_shuffle_i32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _f6 = _mm512_shuffle_i32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 3, 1, 3));
                _f7 = _mm512_shuffle_i32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
            }

            {
                if (out_elempack == 16)
                {
                    _mm512_store_si512((__m512i*)p0, _f0);
                    _mm512_store_si512((__m512i*)(p0 + 16), _f1);
                    _mm512_store_si512((__m512i*)(p0 + 32), _f2);
                    _mm512_store_si512((__m512i*)(p0 + 48), _f3);
                    _mm512_store_si512((__m512i*)(p0 + 64), _f4);
                    _mm512_store_si512((__m512i*)(p0 + 80), _f5);
                    _mm512_store_si512((__m512i*)(p0 + 96), _f6);
                    _mm512_store_si512((__m512i*)(p0 + 112), _f7);
                    p0 += 128;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_si256((__m256i*)p0, _mm512_extracti32x8_epi32(_f0, 0));
                    _mm256_store_si256((__m256i*)(p0 + 8), _mm512_extracti32x8_epi32(_f1, 0));
                    _mm256_store_si256((__m256i*)(p0 + 16), _mm512_extracti32x8_epi32(_f2, 0));
                    _mm256_store_si256((__m256i*)(p0 + 24), _mm512_extracti32x8_epi32(_f3, 0));
                    _mm256_store_si256((__m256i*)(p0 + 32), _mm512_extracti32x8_epi32(_f4, 0));
                    _mm256_store_si256((__m256i*)(p0 + 40), _mm512_extracti32x8_epi32(_f5, 0));
                    _mm256_store_si256((__m256i*)(p0 + 48), _mm512_extracti32x8_epi32(_f6, 0));
                    _mm256_store_si256((__m256i*)(p0 + 56), _mm512_extracti32x8_epi32(_f7, 0));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8), _mm512_extracti32x8_epi32(_f0, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 8), _mm512_extracti32x8_epi32(_f1, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 16), _mm512_extracti32x8_epi32(_f2, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 24), _mm512_extracti32x8_epi32(_f3, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 32), _mm512_extracti32x8_epi32(_f4, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 40), _mm512_extracti32x8_epi32(_f5, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 48), _mm512_extracti32x8_epi32(_f6, 1));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8 + 56), _mm512_extracti32x8_epi32(_f7, 1));
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f4 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f5 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _f6 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _f7 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)p0, _f0);
                    _mm512_storeu_si512((__m512i*)(p0 + 16), _f1);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4), _f2);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4 + 16), _f3);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8), _f4);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8 + 16), _f5);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 12), _f6);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 12 + 16), _f7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose16x8_epi32(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm256_storeu_si256((__m256i*)p0, _mm512_extracti32x8_epi32(_f0, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _mm512_extracti32x8_epi32(_f0, 1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), _mm512_extracti32x8_epi32(_f1, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), _mm512_extracti32x8_epi32(_f1, 1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), _mm512_extracti32x8_epi32(_f2, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), _mm512_extracti32x8_epi32(_f2, 1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), _mm512_extracti32x8_epi32(_f3, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), _mm512_extracti32x8_epi32(_f3, 1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 8), _mm512_extracti32x8_epi32(_f4, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 9), _mm512_extracti32x8_epi32(_f4, 1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 10), _mm512_extracti32x8_epi32(_f5, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 11), _mm512_extracti32x8_epi32(_f5, 1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 12), _mm512_extracti32x8_epi32(_f6, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 13), _mm512_extracti32x8_epi32(_f6, 1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 14), _mm512_extracti32x8_epi32(_f7, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 15), _mm512_extracti32x8_epi32(_f7, 1));
                    p0 += 8;
                }
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512i _f0 = _mm512_load_si512((const __m512i*)pp);
            __m512i _f1 = _mm512_load_si512((const __m512i*)(pp + 16));
            __m512i _f2 = _mm512_load_si512((const __m512i*)(pp + 32));
            __m512i _f3 = _mm512_load_si512((const __m512i*)(pp + 48));
            pp += 64;

            // from
            //      00 11 22 33 40 51 62 73 80 91 a2 b3 c0 d1 e2 f3
            //      01 12 23 30 41 52 63 70 81 92 a3 b0 c1 d2 e3 f0
            //      20 31 02 13 60 71 42 53 a0 b1 82 93 e0 f1 c2 d3
            //      21 32 03 10 61 72 43 50 a1 b2 83 90 e1 f2 c3 d0
            // to
            //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
            //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
            //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
            //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
            {
                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
                __m512i _tmp0 = _mm512_unpacklo_epi32(_f0, _f3);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_f0, _f3);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_f2, _f1);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_f2, _f1);
                _f0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _f1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _f2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                _f3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
            }

            {
                if (out_elempack == 16)
                {
                    _mm512_store_si512((__m512i*)p0, _f0);
                    _mm512_store_si512((__m512i*)(p0 + 16), _f1);
                    _mm512_store_si512((__m512i*)(p0 + 32), _f2);
                    _mm512_store_si512((__m512i*)(p0 + 48), _f3);
                    p0 += 64;
                }
                if (out_elempack == 8)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_si512((__m512i*)p0, _tmp0);
                    _mm512_storeu_si512((__m512i*)(p0 + 16), _tmp1);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8), _tmp2);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8 + 16), _tmp3);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f2 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)p0, _f0);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4), _f1);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8), _f2);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 12), _f3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose16x4_epi32(_f0, _f1, _f2, _f3);

                    _mm_storeu_si128((__m128i*)p0, _mm512_extracti32x4_epi32(_f0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _mm512_extracti32x4_epi32(_f0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _mm512_extracti32x4_epi32(_f0, 2));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _mm512_extracti32x4_epi32(_f0, 3));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 4), _mm512_extracti32x4_epi32(_f1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 5), _mm512_extracti32x4_epi32(_f1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 6), _mm512_extracti32x4_epi32(_f1, 2));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 7), _mm512_extracti32x4_epi32(_f1, 3));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm512_extracti32x4_epi32(_f2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 9), _mm512_extracti32x4_epi32(_f2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 10), _mm512_extracti32x4_epi32(_f2, 2));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 11), _mm512_extracti32x4_epi32(_f2, 3));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 12), _mm512_extracti32x4_epi32(_f3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 13), _mm512_extracti32x4_epi32(_f3, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 14), _mm512_extracti32x4_epi32(_f3, 2));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 15), _mm512_extracti32x4_epi32(_f3, 3));
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512i _f0 = _mm512_load_si512((const __m512i*)pp);
            __m512i _f1 = _mm512_load_si512((const __m512i*)(pp + 16));
            pp += 32;

            // from
            //      00 11 20 31 40 51 60 71 80 91 a0 b1 c0 d1 e0 f1
            //      01 10 21 30 41 50 61 70 81 90 a1 b0 c1 d0 e1 f0
            // to
            //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
            //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
            {
                __m512i _tmp0 = _mm512_shuffle_epi32(_f0, _MM_PERM_DBCA);
                __m512i _tmp1 = _mm512_shuffle_epi32(_f1, _MM_PERM_ACDB);
                _f0 = _mm512_unpacklo_epi32(_tmp0, _tmp1);
                _f1 = _mm512_unpackhi_epi32(_tmp0, _tmp1);
                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
            }

            {
                if (out_elempack == 16)
                {
                    _mm512_store_si512((__m512i*)p0, _f0);
                    _mm512_store_si512((__m512i*)(p0 + 16), _f1);
                    p0 += 32;
                }
                if (out_elempack == 8)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    _mm512_storeu_si512((__m512i*)p0, _tmp0);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 8), _tmp1);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)p0, _mm512_extracti32x4_epi32(_f0, 0));
                    _mm_store_si128((__m128i*)(p0 + 4), _mm512_extracti32x4_epi32(_f1, 0));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 4), _mm512_extracti32x4_epi32(_f0, 1));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 4 + 4), _mm512_extracti32x4_epi32(_f1, 1));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 8), _mm512_extracti32x4_epi32(_f0, 2));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 8 + 4), _mm512_extracti32x4_epi32(_f1, 2));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 12), _mm512_extracti32x4_epi32(_f0, 3));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 12 + 4), _mm512_extracti32x4_epi32(_f1, 3));
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_epi32(p0, _vindex, _f0, sizeof(int));
                    _mm512_i32scatter_epi32(p0 + 1, _vindex, _f1, sizeof(int));
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m512i _f0 = _mm512_load_si512((const __m512i*)pp);
            pp += 16;

            {
                if (out_elempack == 16)
                {
                    _mm512_store_si512((__m512i*)p0, _f0);
                    p0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_si256((__m256i*)p0, _mm512_extracti32x8_epi32(_f0, 0));
                    _mm256_store_si256((__m256i*)(p0 + out_hstep * 8), _mm512_extracti32x8_epi32(_f0, 1));
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)p0, _mm512_extracti32x4_epi32(_f0, 0));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 4), _mm512_extracti32x4_epi32(_f0, 1));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 8), _mm512_extracti32x4_epi32(_f0, 2));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 12), _mm512_extracti32x4_epi32(_f0, 3));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_epi32(p0, _vindex, _f0, sizeof(int));
                    p0++;
                }
            }
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    const int* pp1 = pp + max_jj * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _f0 = _mm512_load_si512((const __m512i*)pp);
            __m512i _f1 = _mm512_load_si512((const __m512i*)(pp + 16));
            __m512i _f2 = _mm512_load_si512((const __m512i*)(pp + 32));
            __m512i _f3 = _mm512_load_si512((const __m512i*)(pp + 48));
            __m512i _f4 = _mm512_load_si512((const __m512i*)(pp + 64));
            __m512i _f5 = _mm512_load_si512((const __m512i*)(pp + 80));
            __m512i _f6 = _mm512_load_si512((const __m512i*)(pp + 96));
            __m512i _f7 = _mm512_load_si512((const __m512i*)(pp + 112));
            pp += 128;

            // from
            //      00 11 22 33  44 55 66 77  08 19 2a 3b  4c 5d 6e 7f
            //      01 12 23 30  45 56 67 74  09 1a 2b 38  4d 5e 6f 7c
            //      20 31 02 13  64 75 46 57  28 39 0a 1b  6c 7d 4e 5f
            //      21 32 03 10  65 76 47 54  29 3a 0b 18  6d 7e 4f 5c
            //      04 15 26 37  40 51 62 73  0c 1d 2e 3f  48 59 6a 7b
            //      05 16 27 34  41 52 63 70  0d 1e 2f 3c  49 5a 6b 78
            //      24 35 06 17  60 71 42 53  2c 3d 0e 1f  68 79 4a 5b
            //      25 36 07 14  61 72 43 50  2d 3e 0f 1c  69 7a 4b 58

            // to
            //      00 10 20 30  44 54 64 74  08 18 28 38  4c 5c 6c 7c
            //      01 11 21 31  45 55 65 75  09 19 29 39  4d 5d 6d 7d
            //      02 12 22 32  46 56 66 76  0a 1a 2a 3a  4e 5e 6e 7e
            //      03 13 23 33  47 57 67 77  0b 1b 2b 3b  4f 5f 6f 7f
            //      04 14 24 34  40 50 60 70  0c 1c 2c 3c  48 58 68 78
            //      05 15 25 35  41 51 61 71  0d 1d 2d 3d  49 59 69 79
            //      06 16 26 36  42 52 62 72  0e 1e 2e 3e  4a 5a 6a 7a
            //      07 17 27 37  43 53 63 73  0f 1f 2f 3f  4b 5b 6b 7b
            {
                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
                _f5 = _mm512_shuffle_epi32(_f5, _MM_PERM_CBAD);
                _f7 = _mm512_shuffle_epi32(_f7, _MM_PERM_CBAD);

                __m512i _tmp0 = _mm512_unpacklo_epi32(_f0, _f3);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_f0, _f3);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_f2, _f1);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_f2, _f1);
                __m512i _tmp4 = _mm512_unpacklo_epi32(_f4, _f7);
                __m512i _tmp5 = _mm512_unpackhi_epi32(_f4, _f7);
                __m512i _tmp6 = _mm512_unpacklo_epi32(_f6, _f5);
                __m512i _tmp7 = _mm512_unpackhi_epi32(_f6, _f5);

                _f0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _f1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _f2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                _f3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                _f4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                _f5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                _f6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                _f7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);

                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
                _f5 = _mm512_shuffle_epi32(_f5, _MM_PERM_CBAD);
                _f7 = _mm512_shuffle_epi32(_f7, _MM_PERM_CBAD);

                _tmp0 = _mm512_shuffle_i32x4(_f0, _f4, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp1 = _mm512_shuffle_i32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp2 = _mm512_shuffle_i32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_i32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp4 = _mm512_shuffle_i32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp5 = _mm512_shuffle_i32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_i32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp7 = _mm512_shuffle_i32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));

                _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _f1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _f2 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _f3 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _f4 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(1, 3, 1, 3));
                _f5 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(1, 3, 1, 3));
                _f6 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _f7 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
            }

            {
                if (out_elempack == 8)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_si512((__m512i*)p0, _tmp0);
                    _mm512_storeu_si512((__m512i*)(p0 + 16), _tmp1);
                    _mm512_storeu_si512((__m512i*)(p0 + 32), _tmp2);
                    _mm512_storeu_si512((__m512i*)(p0 + 48), _tmp3);
                    _mm512_storeu_si512((__m512i*)(p0 + 64), _tmp4);
                    _mm512_storeu_si512((__m512i*)(p0 + 80), _tmp5);
                    _mm512_storeu_si512((__m512i*)(p0 + 96), _tmp6);
                    _mm512_storeu_si512((__m512i*)(p0 + 112), _tmp7);
                    p0 += 128;
                }
                if (out_elempack == 4)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _f4 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f5 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f6 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _f7 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)p0, _f0);
                    _mm512_storeu_si512((__m512i*)(p0 + 16), _f1);
                    _mm512_storeu_si512((__m512i*)(p0 + 32), _f2);
                    _mm512_storeu_si512((__m512i*)(p0 + 48), _f3);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4), _f4);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4 + 16), _f5);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4 + 32), _f6);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4 + 48), _f7);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512i _tmp0 = _mm512_unpacklo_epi32(_f0, _f1);
                    __m512i _tmp1 = _mm512_unpacklo_epi32(_f2, _f3);
                    __m512i _tmp2 = _mm512_unpacklo_epi32(_f4, _f5);
                    __m512i _tmp3 = _mm512_unpacklo_epi32(_f6, _f7);
                    __m512i _tmp4 = _mm512_unpackhi_epi32(_f0, _f1);
                    __m512i _tmp5 = _mm512_unpackhi_epi32(_f2, _f3);
                    __m512i _tmp6 = _mm512_unpackhi_epi32(_f4, _f5);
                    __m512i _tmp7 = _mm512_unpackhi_epi32(_f6, _f7);

                    _f0 = _mm512_unpacklo_epi64(_tmp0, _tmp1);
                    _f1 = _mm512_unpacklo_epi64(_tmp2, _tmp3);
                    _f2 = _mm512_unpackhi_epi64(_tmp0, _tmp1);
                    _f3 = _mm512_unpackhi_epi64(_tmp2, _tmp3);
                    _f4 = _mm512_unpacklo_epi64(_tmp4, _tmp5);
                    _f5 = _mm512_unpacklo_epi64(_tmp6, _tmp7);
                    _f6 = _mm512_unpackhi_epi64(_tmp4, _tmp5);
                    _f7 = _mm512_unpackhi_epi64(_tmp6, _tmp7);

                    _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp2 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp3 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp4 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp5 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp6 = _mm512_shuffle_i32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp7 = _mm512_shuffle_i32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _f1 = _mm512_shuffle_i32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _f2 = _mm512_shuffle_i32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _f3 = _mm512_shuffle_i32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _f4 = _mm512_shuffle_i32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _f5 = _mm512_shuffle_i32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _f6 = _mm512_shuffle_i32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _f7 = _mm512_shuffle_i32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm512_storeu_si512((__m512i*)p0, _f0);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep), _f1);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 2), _f2);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 3), _f3);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 4), _f4);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 5), _f5);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 6), _f6);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 7), _f7);

                    p0 += 16;
                }
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX2__
            __m256i _f0 = _mm256_load_si256((const __m256i*)pp);
            __m256i _f1 = _mm256_load_si256((const __m256i*)(pp + 8));
            __m256i _f2 = _mm256_load_si256((const __m256i*)(pp + 16));
            __m256i _f3 = _mm256_load_si256((const __m256i*)(pp + 24));
            __m256i _f4 = _mm256_load_si256((const __m256i*)(pp + 32));
            __m256i _f5 = _mm256_load_si256((const __m256i*)(pp + 40));
            __m256i _f6 = _mm256_load_si256((const __m256i*)(pp + 48));
            __m256i _f7 = _mm256_load_si256((const __m256i*)(pp + 56));
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
                __m256i _tmp0 = _f0;
                __m256i _tmp1 = _mm256_shuffle_epi32(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp2 = _f2;
                __m256i _tmp3 = _mm256_shuffle_epi32(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp4 = _f4;
                __m256i _tmp5 = _mm256_shuffle_epi32(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp6 = _f6;
                __m256i _tmp7 = _mm256_shuffle_epi32(_f7, _MM_SHUFFLE(2, 1, 0, 3));

                _f0 = _mm256_unpacklo_epi32(_tmp0, _tmp3);
                _f1 = _mm256_unpackhi_epi32(_tmp0, _tmp3);
                _f2 = _mm256_unpacklo_epi32(_tmp2, _tmp1);
                _f3 = _mm256_unpackhi_epi32(_tmp2, _tmp1);
                _f4 = _mm256_unpacklo_epi32(_tmp4, _tmp7);
                _f5 = _mm256_unpackhi_epi32(_tmp4, _tmp7);
                _f6 = _mm256_unpacklo_epi32(_tmp6, _tmp5);
                _f7 = _mm256_unpackhi_epi32(_tmp6, _tmp5);

                _tmp0 = _mm256_unpacklo_epi64(_f0, _f2);
                _tmp1 = _mm256_unpackhi_epi64(_f0, _f2);
                _tmp2 = _mm256_unpacklo_epi64(_f3, _f1);
                _tmp3 = _mm256_unpackhi_epi64(_f3, _f1);
                _tmp4 = _mm256_unpacklo_epi64(_f4, _f6);
                _tmp5 = _mm256_unpackhi_epi64(_f4, _f6);
                _tmp6 = _mm256_unpacklo_epi64(_f7, _f5);
                _tmp7 = _mm256_unpackhi_epi64(_f7, _f5);

                _tmp1 = _mm256_shuffle_epi32(_tmp1, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp3 = _mm256_shuffle_epi32(_tmp3, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp5 = _mm256_shuffle_epi32(_tmp5, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp7 = _mm256_shuffle_epi32(_tmp7, _MM_SHUFFLE(2, 1, 0, 3));

                _f0 = _mm256_permute2x128_si256(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 0));
                _f1 = _mm256_permute2x128_si256(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 0));
                _f2 = _mm256_permute2x128_si256(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 0));
                _f3 = _mm256_permute2x128_si256(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 0));
                _f4 = _mm256_permute2x128_si256(_tmp4, _tmp0, _MM_SHUFFLE(0, 3, 0, 0));
                _f5 = _mm256_permute2x128_si256(_tmp5, _tmp1, _MM_SHUFFLE(0, 3, 0, 0));
                _f6 = _mm256_permute2x128_si256(_tmp6, _tmp2, _MM_SHUFFLE(0, 3, 0, 0));
                _f7 = _mm256_permute2x128_si256(_tmp7, _tmp3, _MM_SHUFFLE(0, 3, 0, 0));
            }
#else  // __AVX2__
            __m256i _f0 = _mm256_loadu_si256((const __m256i*)pp);
            __m256i _f1 = _mm256_loadu_si256((const __m256i*)(pp + 8));
            __m256i _f2 = _mm256_loadu_si256((const __m256i*)(pp + 16));
            __m256i _f3 = _mm256_loadu_si256((const __m256i*)(pp + 24));
            __m256i _f4 = _mm256_loadu_si256((const __m256i*)pp1);
            __m256i _f5 = _mm256_loadu_si256((const __m256i*)(pp1 + 8));
            __m256i _f6 = _mm256_loadu_si256((const __m256i*)(pp1 + 16));
            __m256i _f7 = _mm256_loadu_si256((const __m256i*)(pp1 + 24));
            pp += 32;
            pp1 += 32;

            // from
            //      00 11 22 33 04 15 26 37
            //      20 31 02 13 24 35 06 17
            //      01 12 23 30 05 16 27 34
            //      21 32 03 10 25 36 07 14
            //      40 51 62 73 44 55 66 77
            //      60 71 42 53 64 75 46 57
            //      41 52 63 70 45 56 67 74
            //      61 72 43 50 65 76 47 54

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
                __m256i _tmp0 = _f0;
                __m256i _tmp1 = _f1;
                __m256i _tmp2 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f2), _mm256_castsi256_ps(_f2), _MM_SHUFFLE(2, 1, 0, 3)));
                __m256i _tmp3 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f3), _mm256_castsi256_ps(_f3), _MM_SHUFFLE(2, 1, 0, 3)));
                __m256i _tmp4 = _f4;
                __m256i _tmp5 = _f5;
                __m256i _tmp6 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f6), _mm256_castsi256_ps(_f6), _MM_SHUFFLE(2, 1, 0, 3)));
                __m256i _tmp7 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f7), _mm256_castsi256_ps(_f7), _MM_SHUFFLE(2, 1, 0, 3)));

                _f0 = _mm256_permute2f128_si256(_tmp0, _tmp4, _MM_SHUFFLE(0, 2, 0, 0));
                _f1 = _mm256_permute2f128_si256(_tmp1, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                _f2 = _mm256_permute2f128_si256(_tmp2, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                _f3 = _mm256_permute2f128_si256(_tmp3, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                _f4 = _mm256_permute2f128_si256(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                _f5 = _mm256_permute2f128_si256(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                _f6 = _mm256_permute2f128_si256(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 1));
                _f7 = _mm256_permute2f128_si256(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                _tmp0 = _mm256_castps_si256(_mm256_unpacklo_ps(_mm256_castsi256_ps(_f0), _mm256_castsi256_ps(_f3)));
                _tmp1 = _mm256_castps_si256(_mm256_unpacklo_ps(_mm256_castsi256_ps(_f1), _mm256_castsi256_ps(_f2)));
                _tmp2 = _mm256_castps_si256(_mm256_unpackhi_ps(_mm256_castsi256_ps(_f1), _mm256_castsi256_ps(_f2)));
                _tmp3 = _mm256_castps_si256(_mm256_unpackhi_ps(_mm256_castsi256_ps(_f0), _mm256_castsi256_ps(_f3)));
                _tmp4 = _mm256_castps_si256(_mm256_unpacklo_ps(_mm256_castsi256_ps(_f4), _mm256_castsi256_ps(_f7)));
                _tmp5 = _mm256_castps_si256(_mm256_unpacklo_ps(_mm256_castsi256_ps(_f5), _mm256_castsi256_ps(_f6)));
                _tmp6 = _mm256_castps_si256(_mm256_unpackhi_ps(_mm256_castsi256_ps(_f5), _mm256_castsi256_ps(_f6)));
                _tmp7 = _mm256_castps_si256(_mm256_unpackhi_ps(_mm256_castsi256_ps(_f4), _mm256_castsi256_ps(_f7)));

                _f0 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(_tmp0), _mm256_castsi256_pd(_tmp1)));
                _f1 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(_tmp0), _mm256_castsi256_pd(_tmp1)));
                _f2 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(_tmp2), _mm256_castsi256_pd(_tmp3)));
                _f3 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(_tmp2), _mm256_castsi256_pd(_tmp3)));
                _f4 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(_tmp4), _mm256_castsi256_pd(_tmp5)));
                _f5 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(_tmp4), _mm256_castsi256_pd(_tmp5)));
                _f6 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(_tmp6), _mm256_castsi256_pd(_tmp7)));
                _f7 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(_tmp6), _mm256_castsi256_pd(_tmp7)));

                _f1 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f1), _mm256_castsi256_ps(_f1), _MM_SHUFFLE(2, 1, 0, 3)));
                _f3 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f3), _mm256_castsi256_ps(_f3), _MM_SHUFFLE(2, 1, 0, 3)));
                _f5 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f5), _mm256_castsi256_ps(_f5), _MM_SHUFFLE(2, 1, 0, 3)));
                _f7 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f7), _mm256_castsi256_ps(_f7), _MM_SHUFFLE(2, 1, 0, 3)));
            }
#endif // __AVX2__

            {
                if (out_elempack == 8)
                {
                    _mm256_store_si256((__m256i*)p0, _f0);
                    _mm256_store_si256((__m256i*)(p0 + 8), _f1);
                    _mm256_store_si256((__m256i*)(p0 + 16), _f2);
                    _mm256_store_si256((__m256i*)(p0 + 24), _f3);
                    _mm256_store_si256((__m256i*)(p0 + 32), _f4);
                    _mm256_store_si256((__m256i*)(p0 + 40), _f5);
                    _mm256_store_si256((__m256i*)(p0 + 48), _f6);
                    _mm256_store_si256((__m256i*)(p0 + 56), _f7);
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m256i _tmp0 = _mm256_permute2f128_si256(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2f128_si256(_f2, _f3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp2 = _mm256_permute2f128_si256(_f4, _f5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp3 = _mm256_permute2f128_si256(_f6, _f7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp4 = _mm256_permute2f128_si256(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp5 = _mm256_permute2f128_si256(_f2, _f3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp6 = _mm256_permute2f128_si256(_f4, _f5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp7 = _mm256_permute2f128_si256(_f6, _f7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_si256((__m256i*)p0, _tmp0);
                    _mm256_storeu_si256((__m256i*)(p0 + 8), _tmp1);
                    _mm256_storeu_si256((__m256i*)(p0 + 16), _tmp2);
                    _mm256_storeu_si256((__m256i*)(p0 + 24), _tmp3);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), _tmp4);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4 + 8), _tmp5);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4 + 16), _tmp6);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4 + 24), _tmp7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_epi32(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm256_storeu_si256((__m256i*)p0, _f0);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _f1);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), _f2);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), _f3);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), _f4);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), _f5);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), _f6);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), _f7);
                    p0 += 8;
                }
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
#if __AVX2__
            __m256i _f0 = _mm256_load_si256((const __m256i*)pp);
            __m256i _f1 = _mm256_load_si256((const __m256i*)(pp + 8));
            __m256i _f2 = _mm256_load_si256((const __m256i*)(pp + 16));
            __m256i _f3 = _mm256_load_si256((const __m256i*)(pp + 24));
            pp += 32;
#else
            __m256i _f01l = _mm256_loadu_si256((const __m256i*)pp);
            __m256i _f23l = _mm256_loadu_si256((const __m256i*)(pp + 8));
            __m256i _f01h = _mm256_loadu_si256((const __m256i*)pp1);
            __m256i _f23h = _mm256_loadu_si256((const __m256i*)(pp1 + 8));
            __m256i _f0 = _mm256_permute2f128_si256(_f01l, _f01h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256i _f1 = _mm256_permute2f128_si256(_f01l, _f01h, _MM_SHUFFLE(0, 3, 0, 1));
            __m256i _f2 = _mm256_permute2f128_si256(_f23l, _f23h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256i _f3 = _mm256_permute2f128_si256(_f23l, _f23h, _MM_SHUFFLE(0, 3, 0, 1));
            pp += 16;
            pp1 += 16;
#endif

            // from
            //      00 11 22 33 40 51 62 73
            //      01 12 23 30 41 52 63 70
            //      20 31 02 13 60 71 42 53
            //      21 32 03 10 61 72 43 50
            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            //      02 12 22 32 42 52 62 72
            //      03 13 23 33 43 53 63 73
            {
#if __AVX2__
                _f1 = _mm256_shuffle_epi32(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_epi32(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp0 = _mm256_unpacklo_epi32(_f0, _f3);
                __m256i _tmp1 = _mm256_unpackhi_epi32(_f0, _f3);
                __m256i _tmp2 = _mm256_unpacklo_epi32(_f2, _f1);
                __m256i _tmp3 = _mm256_unpackhi_epi32(_f2, _f1);
                _f0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                _f1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                _f2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                _f3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                _f1 = _mm256_shuffle_epi32(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_epi32(_f3, _MM_SHUFFLE(2, 1, 0, 3));
#else
                _f1 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f1), _mm256_castsi256_ps(_f1), _MM_SHUFFLE(2, 1, 0, 3)));
                _f3 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f3), _mm256_castsi256_ps(_f3), _MM_SHUFFLE(2, 1, 0, 3)));
                __m256 _tmp0 = _mm256_unpacklo_ps(_mm256_castsi256_ps(_f0), _mm256_castsi256_ps(_f3));
                __m256 _tmp1 = _mm256_unpackhi_ps(_mm256_castsi256_ps(_f0), _mm256_castsi256_ps(_f3));
                __m256 _tmp2 = _mm256_unpacklo_ps(_mm256_castsi256_ps(_f2), _mm256_castsi256_ps(_f1));
                __m256 _tmp3 = _mm256_unpackhi_ps(_mm256_castsi256_ps(_f2), _mm256_castsi256_ps(_f1));
                _f0 = _mm256_castps_si256(_mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2))));
                _f1 = _mm256_castps_si256(_mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2))));
                _f2 = _mm256_castps_si256(_mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1))));
                _f3 = _mm256_castps_si256(_mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1))));
                _f1 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f1), _mm256_castsi256_ps(_f1), _MM_SHUFFLE(2, 1, 0, 3)));
                _f3 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f3), _mm256_castsi256_ps(_f3), _MM_SHUFFLE(2, 1, 0, 3)));
#endif
            }

            {
                if (out_elempack == 8)
                {
                    _mm256_store_si256((__m256i*)p0, _f0);
                    _mm256_store_si256((__m256i*)(p0 + 8), _f1);
                    _mm256_store_si256((__m256i*)(p0 + 16), _f2);
                    _mm256_store_si256((__m256i*)(p0 + 24), _f3);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m256i _tmp0 = _mm256_permute2f128_si256(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2f128_si256(_f2, _f3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp2 = _mm256_permute2f128_si256(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp3 = _mm256_permute2f128_si256(_f2, _f3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_si256((__m256i*)p0, _tmp0);
                    _mm256_storeu_si256((__m256i*)(p0 + 8), _tmp1);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), _tmp2);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4 + 8), _tmp3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_epi32(_f0, _f1, _f2, _f3);
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_f0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _mm256_extractf128_si256(_f0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _mm256_extractf128_si256(_f1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _mm256_extractf128_si256(_f1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 4), _mm256_extractf128_si256(_f2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 5), _mm256_extractf128_si256(_f2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 6), _mm256_extractf128_si256(_f3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 7), _mm256_extractf128_si256(_f3, 1));
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __AVX2__
            __m256i _f0 = _mm256_load_si256((const __m256i*)pp);
            __m256i _f1 = _mm256_load_si256((const __m256i*)(pp + 8));
            pp += 16;
#else
            __m256i _f01l = _mm256_loadu_si256((const __m256i*)pp);
            __m256i _f01h = _mm256_loadu_si256((const __m256i*)pp1);
            __m256i _f0 = _mm256_permute2f128_si256(_f01l, _f01h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256i _f1 = _mm256_permute2f128_si256(_f01l, _f01h, _MM_SHUFFLE(0, 3, 0, 1));
            pp += 8;
            pp1 += 8;
#endif

            // from
            //      00 11 20 31 40 51 60 71
            //      01 10 21 30 41 50 61 70
            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            {
#if __AVX2__
                __m256i _tmp0 = _mm256_shuffle_epi32(_f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m256i _tmp1 = _mm256_shuffle_epi32(_f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm256_unpacklo_epi32(_tmp0, _tmp1);
                _f1 = _mm256_unpackhi_epi32(_tmp0, _tmp1);
                _f1 = _mm256_shuffle_epi32(_f1, _MM_SHUFFLE(2, 1, 0, 3));
#else
                __m256 _tmp0 = _mm256_shuffle_ps(_mm256_castsi256_ps(_f0), _mm256_castsi256_ps(_f0), _MM_SHUFFLE(3, 1, 2, 0));
                __m256 _tmp1 = _mm256_shuffle_ps(_mm256_castsi256_ps(_f1), _mm256_castsi256_ps(_f1), _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm256_castps_si256(_mm256_unpacklo_ps(_tmp0, _tmp1));
                _f1 = _mm256_castps_si256(_mm256_unpackhi_ps(_tmp0, _tmp1));
                _f1 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(_f1), _mm256_castsi256_ps(_f1), _MM_SHUFFLE(2, 1, 0, 3)));
#endif
            }

            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)p0, _f0);
                    _mm256_storeu_si256((__m256i*)(p0 + 8), _f1);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m256i _tmp0 = _mm256_permute2f128_si256(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2f128_si256(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_si256((__m256i*)p0, _tmp0);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), _tmp1);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(out_hstep));
                    _mm256_i32scatter_epi32(p0, _vindex, _f0, sizeof(int));
                    _mm256_i32scatter_epi32(p0 + 1, _vindex, _f1, sizeof(int));
#else
                    int sum0[8];
                    int sum1[8];
                    _mm256_storeu_si256((__m256i*)sum0, _f0);
                    _mm256_storeu_si256((__m256i*)sum1, _f1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];
                    p0[out_hstep * 4] = sum0[4];
                    p0[out_hstep * 4 + 1] = sum1[4];
                    p0[out_hstep * 5] = sum0[5];
                    p0[out_hstep * 5 + 1] = sum1[5];
                    p0[out_hstep * 6] = sum0[6];
                    p0[out_hstep * 6 + 1] = sum1[6];
                    p0[out_hstep * 7] = sum0[7];
                    p0[out_hstep * 7 + 1] = sum1[7];
#endif // __AVX512F__
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
#if __AVX2__
            __m256i _f0 = _mm256_load_si256((const __m256i*)pp);
            pp += 8;
#else
            __m128i _f0l = _mm_load_si128((const __m128i*)pp);
            __m128i _f0h = _mm_load_si128((const __m128i*)pp1);
            __m256i _f0 = combine4x2_epi32(_f0l, _f0h);
            pp += 4;
            pp1 += 4;
#endif

            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)p0, _f0);
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)p0, _mm256_extractf128_si256(_f0, 0));
                    _mm_store_si128((__m128i*)(p0 + out_hstep * 4), _mm256_extractf128_si256(_f0, 1));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(out_hstep));
                    _mm256_i32scatter_epi32(p0, _vindex, _f0, sizeof(int));
#else
                    int sum0[8];
                    _mm256_storeu_si256((__m256i*)sum0, _f0);
                    p0[0] = sum0[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 4] = sum0[4];
                    p0[out_hstep * 5] = sum0[5];
                    p0[out_hstep * 6] = sum0[6];
                    p0[out_hstep * 7] = sum0[7];
#endif // __AVX512F__
                    p0++;
                }
            }
        }

#if !__AVX2__
        pp = pp1;
        pp1 = pp + max_jj * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _f0 = _mm512_loadu_si512((const __m512i*)pp);
            __m512i _f1 = _mm512_loadu_si512((const __m512i*)(pp + 16));
            __m512i _f2 = _mm512_loadu_si512((const __m512i*)(pp + 32));
            __m512i _f3 = _mm512_loadu_si512((const __m512i*)(pp + 48));

            // from
            //      00 11 22 33 04 15 26 37 08 19 2a 3b 0c 1d 2e 3f
            //      01 12 23 30 05 16 27 34 09 1a 2b 38 0d 1e 2f 3c
            //      20 31 02 13 24 35 06 17 28 3a 0a 1b 2c 3d 0e 1f
            //      21 32 03 10 25 36 07 14 29 3a 0b 18 2d 3e 0f 1c
            // to
            //      00 10 20 30 04 14 24 34 08 18 28 38 0c 1c 2c 3c
            //      01 11 21 31 05 15 25 35 09 19 29 39 0d 1d 2d 3d
            //      02 12 22 32 06 16 26 36 0a 1a 2a 3a 0e 1e 2e 3e
            //      03 13 23 33 07 17 27 37 0b 1b 2b 3b 0f 1f 2f 3f
            {
                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
                __m512i _tmp0 = _mm512_unpacklo_epi32(_f0, _f3);
                __m512i _tmp1 = _mm512_unpacklo_epi32(_f2, _f1);
                __m512i _tmp2 = _mm512_unpackhi_epi32(_f0, _f3);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_f2, _f1);
                _f0 = _mm512_unpacklo_epi64(_tmp0, _tmp1);
                _f1 = _mm512_unpackhi_epi64(_tmp0, _tmp1);
                _f2 = _mm512_unpacklo_epi64(_tmp3, _tmp2);
                _f3 = _mm512_unpackhi_epi64(_tmp3, _tmp2);
                _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);
                _f3 = _mm512_shuffle_epi32(_f3, _MM_PERM_CBAD);
            }

            {
                if (out_elempack == 4)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    _f0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f2 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)p0, _f0);
                    _mm512_storeu_si512((__m512i*)(p0 + 16), _f1);
                    _mm512_storeu_si512((__m512i*)(p0 + 32), _f2);
                    _mm512_storeu_si512((__m512i*)(p0 + 48), _f3);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512i _tmp0 = _mm512_unpacklo_epi32(_f0, _f1);
                    __m512i _tmp1 = _mm512_unpacklo_epi32(_f2, _f3);
                    __m512i _tmp2 = _mm512_unpackhi_epi32(_f0, _f1);
                    __m512i _tmp3 = _mm512_unpackhi_epi32(_f2, _f3);
                    _f0 = _mm512_unpacklo_epi64(_tmp0, _tmp1);
                    _f1 = _mm512_unpackhi_epi64(_tmp0, _tmp1);
                    _f2 = _mm512_unpacklo_epi64(_tmp2, _tmp3);
                    _f3 = _mm512_unpackhi_epi64(_tmp2, _tmp3);

                    _mm512_storeu_si512((__m512i*)p0, _f0);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep), _f1);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 2), _f2);
                    _mm512_storeu_si512((__m512i*)(p0 + out_hstep * 3), _f3);
                    p0 += 16;
                }
            }

            pp += 64;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _f0 = _mm_load_si128((const __m128i*)pp);
            __m128i _f1 = _mm_load_si128((const __m128i*)(pp + 4));
            __m128i _f2 = _mm_load_si128((const __m128i*)(pp + 8));
            __m128i _f3 = _mm_load_si128((const __m128i*)(pp + 12));
            __m128i _f4 = _mm_load_si128((const __m128i*)(pp + 16));
            __m128i _f5 = _mm_load_si128((const __m128i*)(pp + 20));
            __m128i _f6 = _mm_load_si128((const __m128i*)(pp + 24));
            __m128i _f7 = _mm_load_si128((const __m128i*)(pp + 28));

            // from
            //      00 11 22 33
            //      04 15 26 37
            //      20 31 02 13
            //      24 35 06 17
            //      01 12 23 30
            //      05 16 27 34
            //      21 32 03 10
            //      25 36 07 14
            // to
            //      00 10 20 30
            //      01 11 21 31
            //      02 12 22 32
            //      03 13 23 33
            //      04 14 24 34
            //      05 15 25 35
            //      06 16 26 36
            //      07 17 27 37
            {
                _f4 = _mm_shuffle_epi32(_f4, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm_shuffle_epi32(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f6 = _mm_shuffle_epi32(_f6, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm_shuffle_epi32(_f7, _MM_SHUFFLE(2, 1, 0, 3));
                __m128i _tmp0 = _mm_unpacklo_epi32(_f0, _f6);
                __m128i _tmp1 = _mm_unpackhi_epi32(_f0, _f6);
                __m128i _tmp2 = _mm_unpacklo_epi32(_f1, _f7);
                __m128i _tmp3 = _mm_unpackhi_epi32(_f1, _f7);
                __m128i _tmp4 = _mm_unpacklo_epi32(_f2, _f4);
                __m128i _tmp5 = _mm_unpackhi_epi32(_f2, _f4);
                __m128i _tmp6 = _mm_unpacklo_epi32(_f3, _f5);
                __m128i _tmp7 = _mm_unpackhi_epi32(_f3, _f5);
                _f0 = _mm_unpacklo_epi64(_tmp0, _tmp4);
                _f1 = _mm_unpackhi_epi64(_tmp0, _tmp4);
                _f2 = _mm_unpacklo_epi64(_tmp5, _tmp1);
                _f3 = _mm_unpackhi_epi64(_tmp5, _tmp1);
                _f4 = _mm_unpacklo_epi64(_tmp2, _tmp6);
                _f5 = _mm_unpackhi_epi64(_tmp2, _tmp6);
                _f6 = _mm_unpacklo_epi64(_tmp7, _tmp3);
                _f7 = _mm_unpackhi_epi64(_tmp7, _tmp3);
                _f1 = _mm_shuffle_epi32(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_epi32(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm_shuffle_epi32(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm_shuffle_epi32(_f7, _MM_SHUFFLE(2, 1, 0, 3));
            }

            {
                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)p0, _f0);
                    _mm_store_si128((__m128i*)(p0 + 4), _f1);
                    _mm_store_si128((__m128i*)(p0 + 8), _f2);
                    _mm_store_si128((__m128i*)(p0 + 12), _f3);
                    _mm_store_si128((__m128i*)(p0 + 16), _f4);
                    _mm_store_si128((__m128i*)(p0 + 20), _f5);
                    _mm_store_si128((__m128i*)(p0 + 24), _f6);
                    _mm_store_si128((__m128i*)(p0 + 28), _f7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi32(_f0, _f1, _f2, _f3);
                    transpose4x4_epi32(_f4, _f5, _f6, _f7);
                    _mm_storeu_si128((__m128i*)p0, _f0);
                    _mm_storeu_si128((__m128i*)(p0 + 4), _f4);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _f1);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep + 4), _f5);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _f2);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2 + 4), _f6);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _f3);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3 + 4), _f7);
                    p0 += 8;
                }
            }

            pp += 32;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _f0 = _mm_load_si128((const __m128i*)pp);
            __m128i _f1 = _mm_load_si128((const __m128i*)(pp + 4));
            __m128i _f2 = _mm_load_si128((const __m128i*)(pp + 8));
            __m128i _f3 = _mm_load_si128((const __m128i*)(pp + 12));

            // from
            //      00 11 22 33
            //      01 12 23 30
            //      20 31 02 13
            //      21 32 03 10
            // to
            //      00 10 20 30
            //      01 11 21 31
            //      02 12 22 32
            //      03 13 23 33
            {
                _f1 = _mm_shuffle_epi32(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_epi32(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m128i _tmp0 = _mm_unpacklo_epi32(_f0, _f3);
                __m128i _tmp1 = _mm_unpackhi_epi32(_f0, _f3);
                __m128i _tmp2 = _mm_unpacklo_epi32(_f2, _f1);
                __m128i _tmp3 = _mm_unpackhi_epi32(_f2, _f1);
                _f0 = _mm_unpacklo_epi64(_tmp0, _tmp2);
                _f1 = _mm_unpackhi_epi64(_tmp0, _tmp2);
                _f2 = _mm_unpacklo_epi64(_tmp3, _tmp1);
                _f3 = _mm_unpackhi_epi64(_tmp3, _tmp1);
                _f1 = _mm_shuffle_epi32(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_epi32(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            {
                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)p0, _f0);
                    _mm_store_si128((__m128i*)(p0 + 4), _f1);
                    _mm_store_si128((__m128i*)(p0 + 8), _f2);
                    _mm_store_si128((__m128i*)(p0 + 12), _f3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi32(_f0, _f1, _f2, _f3);
                    _mm_storeu_si128((__m128i*)p0, _f0);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _f1);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _f2);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _f3);
                    p0 += 4;
                }
            }

            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _f0 = _mm_load_si128((const __m128i*)pp);
            __m128i _f1 = _mm_load_si128((const __m128i*)(pp + 4));

            // from
            //      00 11 20 31
            //      01 10 21 30
            // to
            //      00 10 20 30
            //      01 11 21 31
            {
                __m128i _tmp0 = _mm_shuffle_epi32(_f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m128i _tmp1 = _mm_shuffle_epi32(_f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm_unpacklo_epi32(_tmp0, _tmp1);
                _f1 = _mm_unpackhi_epi32(_tmp0, _tmp1);
                _f1 = _mm_shuffle_epi32(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            {
                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)p0, _f0);
                    _mm_store_si128((__m128i*)(p0 + 4), _f1);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m128i _vindex = _mm_mullo_epi32(_mm_setr_epi32(0, 1, 2, 3), _mm_set1_epi32(out_hstep));
                    _mm_i32scatter_epi32(p0, _vindex, _f0, sizeof(int));
                    _mm_i32scatter_epi32(p0 + 1, _vindex, _f1, sizeof(int));
#else
                    int sum0[4];
                    int sum1[4];
                    _mm_storeu_si128((__m128i*)sum0, _f0);
                    _mm_storeu_si128((__m128i*)sum1, _f1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];
#endif // __AVX512F__
                    p0 += 2;
                }
            }

            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _f0 = _mm_load_si128((const __m128i*)pp);

            {
                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)p0, _f0);
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m128i _vindex = _mm_mullo_epi32(_mm_setr_epi32(0, 1, 2, 3), _mm_set1_epi32(out_hstep));
                    _mm_i32scatter_epi32(p0, _vindex, _f0, sizeof(int));
#else
                    int sum0[4];
                    _mm_storeu_si128((__m128i*)sum0, _f0);
                    p0[0] = sum0[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 3] = sum0[3];
#endif // __AVX512F__
                    p0++;
                }
            }

            pp += 4;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        int* p0;
        {
            // out_elempack == 1
            p0 = (int*)top_blob + (i + ii) * out_hstep + j;
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _f0 = _mm512_loadu_si512((const __m512i*)pp);
            __m512i _f1 = _mm512_loadu_si512((const __m512i*)(pp + 16));

            // 00 11 02 13  04 15 06 17  08 19 0a 1b  0c 1d 0e 1f
            // 01 12 03 10  05 16 07 14  09 1a 0b 18  0d 1e 0f 1c

            __m512i _tmp0 = _mm512_unpacklo_epi32(_f0, _f1);
            __m512i _tmp1 = _mm512_unpackhi_epi32(_f0, _f1);

            _f0 = _mm512_unpacklo_epi64(_tmp0, _tmp1);
            _f1 = _mm512_unpackhi_epi64(_tmp0, _tmp1);

            _f1 = _mm512_shuffle_epi32(_f1, _MM_PERM_CBAD);

            {
                _mm512_storeu_si512((__m512i*)p0, _f0);
                _mm512_storeu_si512((__m512i*)(p0 + out_hstep), _f1);
                p0 += 16;
            }

            pp += 32;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _f0 = _mm_load_si128((const __m128i*)pp);
            __m128i _f1 = _mm_load_si128((const __m128i*)(pp + 4));
            __m128i _f2 = _mm_load_si128((const __m128i*)(pp + 8));
            __m128i _f3 = _mm_load_si128((const __m128i*)(pp + 12));

            // 00 11 02 13
            // 04 15 06 17
            // 10 01 12 03
            // 14 05 16 07
            _f2 = _mm_shuffle_epi32(_f2, _MM_SHUFFLE(2, 3, 0, 1));
            _f3 = _mm_shuffle_epi32(_f3, _MM_SHUFFLE(2, 3, 0, 1));

            __m128i _tmp0 = _mm_unpacklo_epi32(_f0, _f2);
            __m128i _tmp1 = _mm_unpackhi_epi32(_f0, _f2);
            __m128i _tmp2 = _mm_unpacklo_epi32(_f1, _f3);
            __m128i _tmp3 = _mm_unpackhi_epi32(_f1, _f3);

            _f0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
            _f1 = _mm_unpacklo_epi64(_tmp2, _tmp3);
            _f2 = _mm_unpackhi_epi64(_tmp0, _tmp1);
            _f3 = _mm_unpackhi_epi64(_tmp2, _tmp3);

            _f2 = _mm_shuffle_epi32(_f2, _MM_SHUFFLE(2, 3, 0, 1));
            _f3 = _mm_shuffle_epi32(_f3, _MM_SHUFFLE(2, 3, 0, 1));

            {
                _mm_storeu_si128((__m128i*)p0, _f0);
                _mm_storeu_si128((__m128i*)(p0 + 4), _f1);
                _mm_storeu_si128((__m128i*)(p0 + out_hstep), _f2);
                _mm_storeu_si128((__m128i*)(p0 + out_hstep + 4), _f3);
                p0 += 8;
            }

            pp += 16;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _f0 = _mm_load_si128((const __m128i*)pp);
            __m128i _f1 = _mm_load_si128((const __m128i*)(pp + 4));

            // 00 11 02 13
            // 01 12 03 10
            __m128i _tmp0 = _mm_unpacklo_epi32(_f0, _f1);
            __m128i _tmp1 = _mm_unpackhi_epi32(_f0, _f1);

            _f0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
            _f1 = _mm_unpackhi_epi64(_tmp1, _tmp0);

            _f1 = _mm_shuffle_epi32(_f1, _MM_SHUFFLE(0, 3, 2, 1));

            {
                _mm_storeu_si128((__m128i*)p0, _f0);
                _mm_storeu_si128((__m128i*)(p0 + out_hstep), _f1);
                p0 += 4;
            }

            pp += 8;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            int f00 = pp[0];
            int f01 = pp[1];
            int f10 = pp[2];
            int f11 = pp[3];

            {
                p0[0] = f00;
                p0[1] = f01;
                p0[out_hstep] = f10;
                p0[out_hstep + 1] = f11;
                p0 += 2;
            }

            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            int f0 = pp[0];
            int f1 = pp[1];

            {
                p0[0] = f0;
                p0[out_hstep] = f1;
                p0++;
            }

            pp += 2;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        int* p0;
        {
            // out_elempack == 1
            p0 = (int*)top_blob + (i + ii) * out_hstep + j;
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _f0 = _mm512_loadu_si512((const __m512i*)pp);

            {
                _mm512_storeu_si512((__m512i*)p0, _f0);
                p0 += 16;
            }

            pp += 16;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _f0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _f1 = _mm_loadu_si128((const __m128i*)(pp + 4));

            {
                _mm_storeu_si128((__m128i*)p0, _f0);
                _mm_storeu_si128((__m128i*)(p0 + 4), _f1);
                p0 += 8;
            }

            pp += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _f0 = _mm_loadu_si128((const __m128i*)pp);

            {
                _mm_storeu_si128((__m128i*)p0, _f0);
                p0 += 4;
            }

            pp += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            int f0 = pp[0];
            int f1 = pp[1];

            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += 2;
            }

            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            int f0 = pp[0];

            p0[0] = f0;

            {
                p0++;
            }

            pp += 1;
        }
    }
}

static int convolution_im2col_gemm_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 1u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile_int8(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                convolution_gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

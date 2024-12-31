// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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
void pack_A_tile_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_fp32_to_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_fp32_to_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void gemm_transB_packed_tile_int8_avx512vnni(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_A_tile_int8_avxvnniint8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_avxvnniint8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_int8_avxvnniint8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_int8_avxvnniint8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_int8_avxvnniint8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_avxvnniint8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_fp32_to_int8_avxvnniint8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_fp32_to_int8_avxvnniint8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void gemm_transB_packed_tile_int8_avxvnniint8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_A_tile_int8_avxvnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_avxvnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_int8_avxvnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_int8_avxvnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_int8_avxvnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_avxvnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_fp32_to_int8_avxvnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_fp32_to_int8_avxvnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void gemm_transB_packed_tile_int8_avxvnni(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void pack_A_tile_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_int8_avx2(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_int8_avx2(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_fp32_to_int8_avx2(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_fp32_to_int8_avx2(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void unpack_output_tile_int32_to_fp32_avx2(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose);
void gemm_transB_packed_tile_int8_avx2(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void gemm_transB_packed_tile_int8_xop(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

static void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_A_tile_int8_avx512vnni(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        pack_A_tile_int8_avxvnniint8(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_A_tile_int8_avxvnni(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_A_tile_int8_avx2(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("pack_A_tile_int8");
    // assert A.elempack == 1
    // assert A.dims == 2

    signed char* pp = AT;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;

        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(A.w));

        int kk = 0;
#if __AVX512VNNI__
        __m512i _w_shift = _mm512_setzero_si512();
        __m512i _v127 = _mm512_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m512i _p = _mm512_i32gather_epi32(_vindex, p0, sizeof(signed char));
            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _p);
            _mm512_storeu_si512((__m512i*)pp, _p);
            pp += 64;
            p0 += 4;
        }
        if (max_kk >= 4)
        {
            _mm512_storeu_si512((__m512i*)pp, _w_shift);
            pp += 64;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            __m256i _p = _mm512_cvtepi32_epi16(_mm512_i32gather_epi32(_vindex, p0, sizeof(signed char)));
            _mm256_storeu_si256((__m256i*)pp, _p);
            pp += 32;
            p0 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            __m128i _p = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, p0, sizeof(signed char)));
            _mm_store_si128((__m128i*)pp, _p);
            pp += 16;
            p0++;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;

        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(A.w));

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
        __m256i _w_shift = _mm256_setzero_si256();
        __m256i _v127 = _mm256_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m256i _p = _mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _p);
            _mm256_storeu_si256((__m256i*)pp, _p);
            pp += 32;
            p0 += 4;
        }
        if (max_kk >= 4)
        {
            _mm256_storeu_si256((__m256i*)pp, _w_shift);
            pp += 32;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            __m128i _p = _mm256_comp_cvtepi32_epi16(_mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
#if __AVX512F__
            _mm_store_si128((__m128i*)pp, _p);
#else
            _mm_storeu_si128((__m128i*)pp, _p);
#endif
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
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;

#if __AVX2__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(A.w));
#else
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;
#endif // __AVX2__

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            _mm_storeu_si128((__m128i*)pp, _p);
            pp += 16;
            p0 += 4;
        }
#else  // __AVXVNNIINT8__
        __m128i _w_shift = _mm_setzero_si128();
        __m128i _v127 = _mm_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _p);
            _mm_storeu_si128((__m128i*)pp, _p);
            pp += 16;
            p0 += 4;
        }
        if (max_kk >= 4)
        {
            _mm_storeu_si128((__m128i*)pp, _w_shift);
            pp += 16;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
#if __AVX2__
            __m128i _p = _mm_comp_cvtepi32_epi16(_mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
            _mm_storel_epi64((__m128i*)pp, _p);
            pp += 8;
            p0 += 2;
#else
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
#endif // __AVX2__
        }
        for (; kk < max_kk; kk++)
        {
#if __AVX2__
            __m128i _p = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
            _mm_store_ss((float*)pp, _mm_castsi128_ps(_p));
            pp += 4;
            p0++;
#else
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
#endif // __AVX2__
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;

        int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#else  // __AVXVNNIINT8__
        int w_shift0 = 0;
        int w_shift1 = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            pp += 8;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += 4;
        }
#else  // __AVXVNNIINT8__
        int w_shift = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            w_shift += pp[0];
            w_shift += pp[1];
            w_shift += pp[2];
            w_shift += pp[3];
            pp += 4;
            p0 += 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift * 127;
            pp += 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_pack_A_tile_int8_avx512vnni(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        transpose_pack_A_tile_int8_avxvnniint8(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_pack_A_tile_int8_avxvnni(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_pack_A_tile_int8_avx2(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("transpose_pack_A_tile_int8");
    // assert A.elempack == 1
    // assert A.dims == 2

    const int A_hstep = A.w;

    signed char* pp = AT;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __AVX512VNNI__
        __m512i _w_shift = _mm512_setzero_si512();
        __m512i _v127 = _mm512_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _p1 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep));
            __m128i _p2 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2));
            __m128i _p3 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3));
            transpose16x4_epi8(_p0, _p1, _p2, _p3);
            __m512i _pp = combine4x4_epi32(_p0, _p1, _p2, _p3);
            _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _pp);
            _mm512_storeu_si512((__m512i*)pp, _pp);
            pp += 64;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            _mm512_storeu_si512((__m512i*)pp, _w_shift);
            pp += 64;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            __m128i _p0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _p1 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep));
            __m128i _t0 = _mm_unpacklo_epi8(_p0, _p1);
            __m128i _t1 = _mm_unpackhi_epi8(_p0, _p1);
            _mm_store_si128((__m128i*)pp, _t0);
            _mm_store_si128((__m128i*)(pp + 16), _t1);
            pp += 32;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)p0);
            _mm_store_si128((__m128i*)pp, _p);
            pp += 16;
            p0 += A_hstep;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep));
            __m128i _p2 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2));
            __m128i _p3 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3));
            transpose8x4_epi8(_p0, _p1, _p2, _p3);
            __m256i _pp = combine4x2_epi32(_p0, _p1);
            _mm256_storeu_si256((__m256i*)pp, _pp);
            pp += 32;
            p0 += A_hstep * 4;
        }
#else  // __AVXVNNIINT8__
        __m256i _w_shift = _mm256_setzero_si256();
        __m256i _v127 = _mm256_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep));
            __m128i _p2 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2));
            __m128i _p3 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3));
            transpose8x4_epi8(_p0, _p1, _p2, _p3);
            __m256i _pp = combine4x2_epi32(_p0, _p1);
            _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _pp);
            _mm256_storeu_si256((__m256i*)pp, _pp);
            pp += 32;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            _mm256_storeu_si256((__m256i*)pp, _w_shift);
            pp += 32;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep));
            __m128i _pp = _mm_unpacklo_epi8(_p0, _p1);
#if __AVX512F__
            _mm_store_si128((__m128i*)pp, _pp);
#else
            _mm_storeu_si128((__m128i*)pp, _pp);
#endif
            pp += 16;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            __m128i _p = _mm_loadl_epi64((const __m128i*)p0);
            _mm_storel_epi64((__m128i*)pp, _p);
            pp += 8;
            p0 += A_hstep;
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

#if __AVX512VNNI__ || __AVXVNNI__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(A_hstep));
#endif // __AVX512VNNI__ || __AVXVNNI__

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _pp = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
            _pp = _mm_shuffle_epi8(_pp, _si);
            _mm_storeu_si128((__m128i*)pp, _pp);
            pp += 16;
            p0 += A_hstep * 4;
        }
#else  // __AVXVNNIINT8__
        __m128i _w_shift = _mm_setzero_si128();
        __m128i _v127 = _mm_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _pp = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
            _pp = _mm_shuffle_epi8(_pp, _si);
            _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp);
            _mm_storeu_si128((__m128i*)pp, _pp);
            pp += 16;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            _mm_storeu_si128((__m128i*)pp, _w_shift);
            pp += 16;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[1];
            pp[3] = p0[A_hstep + 1];
            pp[4] = p0[2];
            pp[5] = p0[A_hstep + 2];
            pp[6] = p0[3];
            pp[7] = p0[A_hstep + 3];
            pp += 8;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += A_hstep;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[A_hstep + 1];
            pp[6] = p0[A_hstep * 2 + 1];
            pp[7] = p0[A_hstep * 3 + 1];
            pp += 8;
            p0 += A_hstep * 4;
        }
#else  // __AVXVNNIINT8__
        int w_shift0 = 0;
        int w_shift1 = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[A_hstep + 1];
            pp[6] = p0[A_hstep * 2 + 1];
            pp[7] = p0[A_hstep * 3 + 1];
            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            pp += 8;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            pp += 8;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[1];
            pp[3] = p0[A_hstep + 1];
            pp += 4;
            p0 += A_hstep * 2;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp += 4;
            p0 += A_hstep * 4;
        }
#else  // __AVXVNNIINT8__
        int w_shift = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            w_shift += pp[0];
            w_shift += pp[1];
            w_shift += pp[2];
            w_shift += pp[3];
            pp += 4;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift * 127;
            pp += 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_B_tile_int8_avx512vnni(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        pack_B_tile_int8_avxvnniint8(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_B_tile_int8_avxvnni(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_B_tile_int8_avx2(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;

        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(B.w));

        int kk = 0;
#if __AVX512VNNI__
        __m512i _v127 = _mm512_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m512i _p = _mm512_i32gather_epi32(_vindex, p0, sizeof(signed char));
            _p = _mm512_add_epi8(_p, _v127);
            _mm512_storeu_si512((__m512i*)pp, _p);
            pp += 64;
            p0 += 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            __m256i _p = _mm512_cvtepi32_epi16(_mm512_i32gather_epi32(_vindex, p0, sizeof(signed char)));
            _mm256_storeu_si256((__m256i*)pp, _p);
            pp += 32;
            p0 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            __m128i _p = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, p0, sizeof(signed char)));
            _mm_store_si128((__m128i*)pp, _p);
            pp += 16;
            p0++;
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;

#if __AVX2__
        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(B.w));
#else
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;
        const signed char* p4 = B.row<const signed char>(j + jj + 4) + k;
        const signed char* p5 = B.row<const signed char>(j + jj + 5) + k;
        const signed char* p6 = B.row<const signed char>(j + jj + 6) + k;
        const signed char* p7 = B.row<const signed char>(j + jj + 7) + k;
#endif // __AVX2__

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
#if __AVX2__
            __m128i _p = _mm256_comp_cvtepi32_epi16(_mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
#if __AVX512F__
            _mm_store_si128((__m128i*)pp, _p);
#else
            _mm_storeu_si128((__m128i*)pp, _p);
#endif
            pp += 16;
            p0 += 2;
#else
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
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
#endif // __AVX2__
        }
        for (; kk < max_kk; kk++)
        {
#if __AVX2__
            __m128i _p = _mm256_comp_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
            _mm_storel_epi64((__m128i*)pp, _p);
            pp += 8;
            p0++;
#else
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
            pp += 8;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
#endif // __AVX2__
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;

#if __AVX2__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(B.w));
#else
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;
#endif // __AVX2__

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            _mm_storeu_si128((__m128i*)pp, _p);
            pp += 16;
            p0 += 4;
        }
#else  // __AVXVNNIINT8__
        __m128i _v127 = _mm_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            _p = _mm_add_epi8(_p, _v127);
            _mm_storeu_si128((__m128i*)pp, _p);
            pp += 16;
            p0 += 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
#if __AVX2__
            __m128i _p = _mm_comp_cvtepi32_epi16(_mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
            _mm_storel_epi64((__m128i*)pp, _p);
            pp += 8;
            p0 += 2;
#else
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
#endif // __AVX2__
        }
        for (; kk < max_kk; kk++)
        {
#if __AVX2__
            __m128i _p = _mm_comp_cvtepi32_epi8(_mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char)));
            _mm_store_ss((float*)pp, _mm_castsi128_ps(_p));
            pp += 4;
            p0++;
#else
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
#endif // __AVX2__
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;

        int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#else  // __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[1] + 127;
            pp[2] = p0[2] + 127;
            pp[3] = p0[3] + 127;
            pp[4] = p1[0] + 127;
            pp[5] = p1[1] + 127;
            pp[6] = p1[2] + 127;
            pp[7] = p1[3] + 127;
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += 4;
        }
#else  // __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[1] + 127;
            pp[2] = p0[2] + 127;
            pp[3] = p0[3] + 127;
            pp += 4;
            p0 += 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_pack_B_tile_int8_avx512vnni(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        transpose_pack_B_tile_int8_avxvnniint8(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_pack_B_tile_int8_avxvnni(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_pack_B_tile_int8_avx2(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("transpose_pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    const int B_hstep = B.w;

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __AVX512VNNI__
        __m512i _v127 = _mm512_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _p1 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep));
            __m128i _p2 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep * 2));
            __m128i _p3 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep * 3));
            transpose16x4_epi8(_p0, _p1, _p2, _p3);
            __m512i _pp = combine4x4_epi32(_p0, _p1, _p2, _p3);
            _pp = _mm512_add_epi8(_pp, _v127);
            _mm512_storeu_si512((__m512i*)pp, _pp);
            pp += 64;
            p0 += B_hstep * 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            __m128i _p0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _p1 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep));
            __m128i _t0 = _mm_unpacklo_epi8(_p0, _p1);
            __m128i _t1 = _mm_unpackhi_epi8(_p0, _p1);
            _mm_store_si128((__m128i*)pp, _t0);
            _mm_store_si128((__m128i*)(pp + 16), _t1);
            pp += 32;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)p0);
            _mm_store_si128((__m128i*)pp, _p);
            pp += 16;
            p0 += B_hstep;
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + B_hstep));
            __m128i _p2 = _mm_loadl_epi64((const __m128i*)(p0 + B_hstep * 2));
            __m128i _p3 = _mm_loadl_epi64((const __m128i*)(p0 + B_hstep * 3));
            transpose8x4_epi8(_p0, _p1, _p2, _p3);
            __m256i _pp = combine4x2_epi32(_p0, _p1);
            _mm256_storeu_si256((__m256i*)pp, _pp);
            pp += 32;
            p0 += B_hstep * 4;
        }
#else  // __AVXVNNIINT8__
        __m256i _v127 = _mm256_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + B_hstep));
            __m128i _p2 = _mm_loadl_epi64((const __m128i*)(p0 + B_hstep * 2));
            __m128i _p3 = _mm_loadl_epi64((const __m128i*)(p0 + B_hstep * 3));
            transpose8x4_epi8(_p0, _p1, _p2, _p3);
            __m256i _pp = combine4x2_epi32(_p0, _p1);
            _pp = _mm256_add_epi8(_pp, _v127);
            _mm256_storeu_si256((__m256i*)pp, _pp);
            pp += 32;
            p0 += B_hstep * 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + B_hstep));
            __m128i _pp = _mm_unpacklo_epi8(_p0, _p1);
#if __AVX512F__
            _mm_store_si128((__m128i*)pp, _pp);
#else
            _mm_storeu_si128((__m128i*)pp, _pp);
#endif
            pp += 16;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            __m128i _p = _mm_loadl_epi64((const __m128i*)p0);
            _mm_storel_epi64((__m128i*)pp, _p);
            pp += 8;
            p0 += B_hstep;
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

#if __AVX512VNNI__ || __AVXVNNI__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(B_hstep));
#endif // __AVX512VNNI__ || __AVXVNNI__

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _pp = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
            _pp = _mm_shuffle_epi8(_pp, _si);
            _mm_storeu_si128((__m128i*)pp, _pp);
            pp += 16;
            p0 += B_hstep * 4;
        }
#else  // __AVXVNNIINT8__
        __m128i _v127 = _mm_set1_epi8(127);
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128i _pp = _mm_i32gather_epi32((const int*)p0, _vindex, sizeof(signed char));
            __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
            _pp = _mm_shuffle_epi8(_pp, _si);
            _pp = _mm_add_epi8(_pp, _v127);
            _mm_storeu_si128((__m128i*)pp, _pp);
            pp += 16;
            p0 += B_hstep * 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[1];
            pp[3] = p0[B_hstep + 1];
            pp[4] = p0[2];
            pp[5] = p0[B_hstep + 2];
            pp[6] = p0[3];
            pp[7] = p0[B_hstep + 3];
            pp += 8;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += B_hstep;
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[B_hstep * 2];
            pp[3] = p0[B_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[B_hstep + 1];
            pp[6] = p0[B_hstep * 2 + 1];
            pp[7] = p0[B_hstep * 3 + 1];
            pp += 8;
            p0 += B_hstep * 4;
        }
#else  // __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[B_hstep] + 127;
            pp[2] = p0[B_hstep * 2] + 127;
            pp[3] = p0[B_hstep * 3] + 127;
            pp[4] = p0[1] + 127;
            pp[5] = p0[B_hstep + 1] + 127;
            pp[6] = p0[B_hstep * 2 + 1] + 127;
            pp[7] = p0[B_hstep * 3 + 1] + 127;
            pp += 8;
            p0 += B_hstep * 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[1];
            pp[3] = p0[B_hstep + 1];
            pp += 4;
            p0 += B_hstep * 2;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
#if __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[B_hstep * 2];
            pp[3] = p0[B_hstep * 3];
            pp += 4;
            p0 += B_hstep * 4;
        }
#else  // __AVXVNNIINT8__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[B_hstep] + 127;
            pp[2] = p0[B_hstep * 2] + 127;
            pp[3] = p0[B_hstep * 3] + 127;
            pp += 4;
            p0 += B_hstep * 4;
        }
#endif // __AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void compute_A_tile_fp32_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("compute_A_tile_int8_scales %d %d", max_ii, elempack);

    const float v127_B_scale = 127.f * B_scale;

    float* ps = (float*)scales + i;
    float* pods = (float*)out_descales + i;

    const int max_ii_packed = max_ii / elempack;
    const int size = A.w * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _v127_avx512 = _mm512_set1_ps(127.f);
    __m512 _v127_B_scale_avx512 = _mm512_set1_ps(v127_B_scale);
#endif // __AVX512F__
    __m256 _v127_avx = _mm256_set1_ps(127.f);
    __m256 _v127_B_scale_avx = _mm256_set1_ps(v127_B_scale);
#endif // __AVX__
    __m128 _v127 = _mm_set1_ps(127.f);
    __m128 _v127_B_scale = _mm_set1_ps(v127_B_scale);
#endif // __SSE2__

    for (int ii = 0; ii < max_ii_packed; ii++)
    {
        const float* ptr = (const float*)A + (i + ii * elempack) * A_hstep;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _absmax_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
        __m256 _absmax_avx = _mm256_set1_ps(0.f);
#endif // __AVX__
        __m128 _absmax = _mm_set1_ps(0.f);
#endif // __SSE2__
        float absmax = 0.f;

        int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; kk + 15 < size; kk += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _absmax_avx512 = _mm512_max_ps(_absmax_avx512, abs512_ps(_p));
            ptr += 16;
        }
#endif // __AVX512F__
        for (; kk + 7 < size; kk += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _absmax_avx = _mm256_max_ps(_absmax_avx, abs256_ps(_p));
            ptr += 8;
        }
#endif // __AVX__
        for (; kk + 3 < size; kk += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _absmax = _mm_max_ps(_absmax, abs_ps(_p));
            ptr += 4;
        }
#endif // __SSE2__
        for (; kk < size; kk++)
        {
            absmax = std::max(absmax, (float)fabsf(*ptr));
            ptr++;
        }

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            __m512 _scale = _mm512_div_ps(_v127_avx512, _absmax_avx512);
            __m512 _out_descale = _mm512_div_ps(_absmax_avx512, _v127_B_scale_avx512);
            _mm512_store_ps(ps, _scale);
            _mm512_store_ps(pods, _out_descale);
            ps += 16;
            pods += 16;
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX512F__
            {
                __m256 _absmax0 = _mm512_castps512_ps256(_absmax_avx512);
                __m256 _absmax1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_absmax_avx512), 1));
                _absmax_avx = _mm256_max_ps(_absmax_avx, _absmax0);
                _absmax_avx = _mm256_max_ps(_absmax_avx, _absmax1);
            }
#endif // __AVX512F__

            __m256 _scale = _mm256_div_ps(_v127_avx, _absmax_avx);
            __m256 _out_descale = _mm256_div_ps(_absmax_avx, _v127_B_scale_avx);
            _mm256_store_ps(ps, _scale);
            _mm256_store_ps(pods, _out_descale);
            ps += 8;
            pods += 8;
        }
#endif // __AVX__
        if (elempack == 4)
        {
#if __AVX__
#if __AVX512F__
            {
                __m256 _absmax0 = _mm512_castps512_ps256(_absmax_avx512);
                __m256 _absmax1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_absmax_avx512), 1));
                _absmax_avx = _mm256_max_ps(_absmax_avx, _absmax0);
                _absmax_avx = _mm256_max_ps(_absmax_avx, _absmax1);
            }
#endif // __AVX512F__
            {
                __m128 _absmax0 = _mm256_castps256_ps128(_absmax_avx);
                __m128 _absmax1 = _mm256_extractf128_ps(_absmax_avx, 1);
                _absmax = _mm_max_ps(_absmax, _absmax0);
                _absmax = _mm_max_ps(_absmax, _absmax1);
            }
#endif // __AVX__

            __m128 _scale = _mm_div_ps(_v127, _absmax);
            __m128 _out_descale = _mm_div_ps(_absmax, _v127_B_scale);
            _mm_store_ps(ps, _scale);
            _mm_store_ps(pods, _out_descale);
            ps += 4;
            pods += 4;
        }
#endif // __SSE2__
        if (elempack == 1)
        {
#if __SSE2__
#if __AVX__
#if __AVX512F__
            absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax_avx512));
#endif // __AVX512F__
            absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax_avx));
#endif // __AVX__
            absmax = std::max(absmax, _mm_reduce_max_ps(_absmax));
#endif // __SSE2__

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
}

static void pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_A_tile_fp32_to_int8_avx512vnni(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        pack_A_tile_fp32_to_int8_avxvnniint8(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_A_tile_fp32_to_int8_avxvnni(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_A_tile_fp32_to_int8_avx2(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("pack_A_tile_fp32_to_int8 %d %d %d", max_ii, max_kk, elempack);

    signed char* pp = (signed char*)AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        __m512 _scales = _mm512_load_ps((const float*)scales + i + ii);
#if __AVX512VNNI__
        __m512i _w_shift = _mm512_setzero_si512();
        __m512i _v127 = _mm512_set1_epi8(127);
#endif // __AVX512VNNI__

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scales);
                _p1 = _mm512_mul_ps(_p1, _scales);
                _p2 = _mm512_mul_ps(_p2, _scales);
                _p3 = _mm512_mul_ps(_p3, _scales);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose16x4_epi8(_pp0, _pp1, _pp2, _pp3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _pp);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += 64;
            }
            if (max_kk >= 4)
            {
                _mm512_storeu_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);

                _p0 = _mm512_mul_ps(_p0, _scales);
                _p1 = _mm512_mul_ps(_p1, _scales);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

                // transpose16x2_epi8
                __m128i _t0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += 32;
            }
            for (; kk < max_kk; kk++)
            {
                __m512 _p = _mm512_load_ps(p0);

                _p = _mm512_mul_ps(_p, _scales);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + A_hstep * 8);
                __m512 _p3 = _mm512_loadu_ps(p0 + A_hstep * 8 + 16);

                __m512 _t0 = _mm512_shuffle_f32x4(_p0, _p2, _MM_SHUFFLE(1, 0, 1, 0));
                __m512 _t1 = _mm512_shuffle_f32x4(_p0, _p2, _MM_SHUFFLE(3, 2, 3, 2));
                __m512 _t2 = _mm512_shuffle_f32x4(_p1, _p3, _MM_SHUFFLE(1, 0, 1, 0));
                __m512 _t3 = _mm512_shuffle_f32x4(_p1, _p3, _MM_SHUFFLE(3, 2, 3, 2));

                _t0 = _mm512_mul_ps(_t0, _scales);
                _t1 = _mm512_mul_ps(_t1, _scales);
                _t2 = _mm512_mul_ps(_t2, _scales);
                _t3 = _mm512_mul_ps(_t3, _scales);

                __m128i _pp0 = float2int8_avx512(_t0);
                __m128i _pp1 = float2int8_avx512(_t1);
                __m128i _pp2 = float2int8_avx512(_t2);
                __m128i _pp3 = float2int8_avx512(_t3);

                transpose16x4_epi8(_pp0, _pp1, _pp2, _pp3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _pp);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += 32;
            }
            if (max_kk >= 4)
            {
                _mm512_storeu_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + A_hstep * 8);

                __m512 _t0 = _mm512_shuffle_f32x4(_p0, _p1, _MM_SHUFFLE(1, 0, 1, 0));
                __m512 _t1 = _mm512_shuffle_f32x4(_p0, _p1, _MM_SHUFFLE(3, 2, 3, 2));

                _t0 = _mm512_mul_ps(_t0, _scales);
                _t1 = _mm512_mul_ps(_t1, _scales);

                __m128i _pp0 = float2int8_avx512(_t0);
                __m128i _pp1 = float2int8_avx512(_t1);

                // transpose16x2_epi8
                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_store_si128((__m128i*)pp, _tt0);
                _mm_store_si128((__m128i*)(pp + 16), _tt1);

                pp += 32;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + A_hstep * 8);

                __m512 _p = combine8x2_ps(_p0, _p1);
                _p = _mm512_mul_ps(_p, _scales);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + A_hstep * 4);
                __m512 _p2 = _mm512_loadu_ps(p0 + A_hstep * 8);
                __m512 _p3 = _mm512_loadu_ps(p0 + A_hstep * 12);

                __m512 _t0 = _mm512_shuffle_f32x4(_p0, _p1, _MM_SHUFFLE(1, 0, 1, 0));
                __m512 _t1 = _mm512_shuffle_f32x4(_p0, _p1, _MM_SHUFFLE(3, 2, 3, 2));
                __m512 _t2 = _mm512_shuffle_f32x4(_p2, _p3, _MM_SHUFFLE(1, 0, 1, 0));
                __m512 _t3 = _mm512_shuffle_f32x4(_p2, _p3, _MM_SHUFFLE(3, 2, 3, 2));

                _p0 = _mm512_shuffle_f32x4(_t0, _t2, _MM_SHUFFLE(2, 0, 2, 0));
                _p1 = _mm512_shuffle_f32x4(_t0, _t2, _MM_SHUFFLE(3, 1, 3, 1));
                _p2 = _mm512_shuffle_f32x4(_t1, _t3, _MM_SHUFFLE(2, 0, 2, 0));
                _p3 = _mm512_shuffle_f32x4(_t1, _t3, _MM_SHUFFLE(3, 1, 3, 1));

                _p0 = _mm512_mul_ps(_p0, _scales);
                _p1 = _mm512_mul_ps(_p1, _scales);
                _p2 = _mm512_mul_ps(_p2, _scales);
                _p3 = _mm512_mul_ps(_p3, _scales);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose16x4_epi8(_pp0, _pp1, _pp2, _pp3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _pp);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += 16;
            }
            if (max_kk >= 4)
            {
                _mm512_storeu_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + A_hstep * 4);
                __m256 _p2 = _mm256_loadu_ps(p0 + A_hstep * 8);
                __m256 _p3 = _mm256_loadu_ps(p0 + A_hstep * 12);

                __m512 _p01 = combine8x2_ps(_p0, _p1);
                __m512 _p23 = combine8x2_ps(_p2, _p3);

                __m512 _t0 = _mm512_shuffle_f32x4(_p01, _p23, _MM_SHUFFLE(2, 0, 2, 0));
                __m512 _t1 = _mm512_shuffle_f32x4(_p01, _p23, _MM_SHUFFLE(3, 1, 3, 1));

                _t0 = _mm512_mul_ps(_t0, _scales);
                _t1 = _mm512_mul_ps(_t1, _scales);

                __m128i _pp0 = float2int8_avx512(_t0);
                __m128i _pp1 = float2int8_avx512(_t1);

                // transpose16x2_epi8
                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_store_si128((__m128i*)pp, _tt0);
                _mm_store_si128((__m128i*)(pp + 16), _tt1);

                pp += 32;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + A_hstep * 4);
                __m128 _p2 = _mm_load_ps(p0 + A_hstep * 8);
                __m128 _p3 = _mm_load_ps(p0 + A_hstep * 12);

                __m512 _p = combine4x4_ps(_p0, _p1, _p2, _p3);
                _p = _mm512_mul_ps(_p, _scales);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + A_hstep * 2);
                __m128 _p3 = _mm_loadu_ps(p0 + A_hstep * 3);
                __m128 _p4 = _mm_loadu_ps(p0 + A_hstep * 4);
                __m128 _p5 = _mm_loadu_ps(p0 + A_hstep * 5);
                __m128 _p6 = _mm_loadu_ps(p0 + A_hstep * 6);
                __m128 _p7 = _mm_loadu_ps(p0 + A_hstep * 7);
                __m128 _p8 = _mm_loadu_ps(p0 + A_hstep * 8);
                __m128 _p9 = _mm_loadu_ps(p0 + A_hstep * 9);
                __m128 _pa = _mm_loadu_ps(p0 + A_hstep * 10);
                __m128 _pb = _mm_loadu_ps(p0 + A_hstep * 11);
                __m128 _pc = _mm_loadu_ps(p0 + A_hstep * 12);
                __m128 _pd = _mm_loadu_ps(p0 + A_hstep * 13);
                __m128 _pe = _mm_loadu_ps(p0 + A_hstep * 14);
                __m128 _pf = _mm_loadu_ps(p0 + A_hstep * 15);

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

                _t0 = _mm512_mul_ps(_t0, _scales);
                _t1 = _mm512_mul_ps(_t1, _scales);
                _t2 = _mm512_mul_ps(_t2, _scales);
                _t3 = _mm512_mul_ps(_t3, _scales);

                __m128i _pp0 = float2int8_avx512(_t0);
                __m128i _pp1 = float2int8_avx512(_t1);
                __m128i _pp2 = float2int8_avx512(_t2);
                __m128i _pp3 = float2int8_avx512(_t3);

                transpose16x4_epi8(_pp0, _pp1, _pp2, _pp3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _pp);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += 4;
            }
            if (max_kk >= 4)
            {
                _mm512_storeu_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(A_hstep));

                __m512 _p0 = _mm512_i32gather_ps(_vindex, p0, sizeof(float));
                __m512 _p1 = _mm512_i32gather_ps(_vindex, p0 + 1, sizeof(float));
                _p0 = _mm512_mul_ps(_p0, _scales);
                _p1 = _mm512_mul_ps(_p1, _scales);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

                __m128i _t0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(A_hstep));

                __m512 _p = _mm512_i32gather_ps(_vindex, p0, sizeof(float));
                _p = _mm512_mul_ps(_p, _scales);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0++;
            }
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    signed char* pp1 = pp + max_kk * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        __m256 _scales = _mm256_load_ps((const float*)scales + i + ii);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m256i _w_shift = _mm256_setzero_si256();
        __m256i _v127 = _mm256_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scales);
                _p1 = _mm256_mul_ps(_p1, _scales);
                _p2 = _mm256_mul_ps(_p2, _scales);
                _p3 = _mm256_mul_ps(_p3, _scales);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);

                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi16(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi16(_tt0, _tt1);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm256_storeu_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += 32;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm256_storeu_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);

                _p0 = _mm256_mul_ps(_p0, _scales);
                _p1 = _mm256_mul_ps(_p1, _scales);

                __m128i _pp = float2int8_avx(_p0, _p1);

                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);

#if __AVX2__
#if __AVX512F__
                _mm_store_si128((__m128i*)pp, _pp);
#else
                _mm_storeu_si128((__m128i*)pp, _pp);
#endif
                pp += 16;
#else
                _mm_storel_pd((double*)pp, _mm_castsi128_pd(_pp));
                _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_pp));
                pp += 8;
                pp1 += 8;
#endif
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                __m256 _p = _mm256_load_ps(p0);

                _p = _mm256_mul_ps(_p, _scales);

                int64_t v = float2int8_avx(_p);

#if __AVX2__
                *(int64_t*)pp = v;
                pp += 8;
#else
                *(int32_t*)pp = (int32_t)v;
                *(int32_t*)pp1 = (int32_t)(v >> 32);
                pp += 4;
                pp1 += 4;
#endif
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + 8);
                __m256 _p2 = _mm256_loadu_ps(p0 + A_hstep * 4);
                __m256 _p3 = _mm256_loadu_ps(p0 + A_hstep * 4 + 8);

                __m256 _t0 = _mm256_permute2f128_ps(_p0, _p2, _MM_SHUFFLE(0, 2, 0, 0));
                __m256 _t1 = _mm256_permute2f128_ps(_p0, _p2, _MM_SHUFFLE(0, 3, 0, 1));
                __m256 _t2 = _mm256_permute2f128_ps(_p1, _p3, _MM_SHUFFLE(0, 2, 0, 0));
                __m256 _t3 = _mm256_permute2f128_ps(_p1, _p3, _MM_SHUFFLE(0, 3, 0, 1));

                _t0 = _mm256_mul_ps(_t0, _scales);
                _t1 = _mm256_mul_ps(_t1, _scales);
                _t2 = _mm256_mul_ps(_t2, _scales);
                _t3 = _mm256_mul_ps(_t3, _scales);

                __m128i _pp0 = float2int8_avx(_t0, _t2);
                __m128i _pp1 = float2int8_avx(_t1, _t3);

                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi16(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi16(_tt0, _tt1);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm256_storeu_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += 16;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm256_storeu_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + A_hstep * 4);

                __m256 _t0 = _mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 2, 0, 0));
                __m256 _t1 = _mm256_permute2f128_ps(_p0, _p1, _MM_SHUFFLE(0, 3, 0, 1));

                _t0 = _mm256_mul_ps(_t0, _scales);
                _t1 = _mm256_mul_ps(_t1, _scales);

                __m128i _pp = float2int8_avx(_t0, _t1);

                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);

#if __AVX2__
#if __AVX512F__
                _mm_store_si128((__m128i*)pp, _pp);
#else
                _mm_storeu_si128((__m128i*)pp, _pp);
#endif
                pp += 16;
#else
                _mm_storel_pd((double*)pp, _mm_castsi128_pd(_pp));
                _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_pp));
                pp += 8;
                pp1 += 8;
#endif
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + A_hstep * 4);

                __m256 _p = combine4x2_ps(_p0, _p1);
                _p = _mm256_mul_ps(_p, _scales);

                int64_t v = float2int8_avx(_p);

#if __AVX2__
                *(int64_t*)pp = v;
                pp += 8;
#else
                *(int32_t*)pp = (int32_t)v;
                *(int32_t*)pp1 = (int32_t)(v >> 32);
                pp += 4;
                pp1 += 4;
#endif
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + A_hstep * 2);
                __m128 _p3 = _mm_loadu_ps(p0 + A_hstep * 3);
                __m128 _p4 = _mm_loadu_ps(p0 + A_hstep * 4);
                __m128 _p5 = _mm_loadu_ps(p0 + A_hstep * 5);
                __m128 _p6 = _mm_loadu_ps(p0 + A_hstep * 6);
                __m128 _p7 = _mm_loadu_ps(p0 + A_hstep * 7);

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

                _t0 = _mm256_mul_ps(_t0, _scales);
                _t1 = _mm256_mul_ps(_t1, _scales);
                _t2 = _mm256_mul_ps(_t2, _scales);
                _t3 = _mm256_mul_ps(_t3, _scales);

                __m128i _pp0 = float2int8_avx(_t0, _t2);
                __m128i _pp1 = float2int8_avx(_t1, _t3);

                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi16(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi16(_tt0, _tt1);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm256_storeu_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm256_storeu_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX2__
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(A_hstep));

                __m256 _p0 = _mm256_i32gather_ps(p0, _vindex, sizeof(float));
                __m256 _p1 = _mm256_i32gather_ps(p0 + 1, _vindex, sizeof(float));
#else
                __m256 _p0 = _mm256_setr_ps(p0[0], p0[A_hstep], p0[A_hstep * 2], p0[A_hstep * 3], p0[A_hstep * 4], p0[A_hstep * 5], p0[A_hstep * 6], p0[A_hstep * 7]);
                __m256 _p1 = _mm256_setr_ps(p0[1], p0[A_hstep + 1], p0[A_hstep * 2 + 1], p0[A_hstep * 3 + 1], p0[A_hstep * 4 + 1], p0[A_hstep * 5 + 1], p0[A_hstep * 6 + 1], p0[A_hstep * 7 + 1]);
#endif

                _p0 = _mm256_mul_ps(_p0, _scales);
                _p1 = _mm256_mul_ps(_p1, _scales);

                __m128i _pp = float2int8_avx(_p0, _p1);

                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);

#if __AVX2__
#if __AVX512F__
                _mm_store_si128((__m128i*)pp, _pp);
#else
                _mm_storeu_si128((__m128i*)pp, _pp);
#endif
                pp += 16;
#else
                _mm_storel_pd((double*)pp, _mm_castsi128_pd(_pp));
                _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_pp));
                pp += 8;
                pp1 += 8;
#endif
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
#if __AVX2__
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(A_hstep));

                __m256 _p = _mm256_i32gather_ps(p0, _vindex, sizeof(float));
#else
                __m256 _p = _mm256_setr_ps(p0[0], p0[A_hstep], p0[A_hstep * 2], p0[A_hstep * 3], p0[A_hstep * 4], p0[A_hstep * 5], p0[A_hstep * 6], p0[A_hstep * 7]);
#endif

                _p = _mm256_mul_ps(_p, _scales);

                int64_t v = float2int8_avx(_p);

#if __AVX2__
                *(int64_t*)pp = v;
                pp += 8;
#else
                *(int32_t*)pp = (int32_t)v;
                *(int32_t*)pp1 = (int32_t)(v >> 32);
                pp += 4;
                pp1 += 4;
#endif
                p0++;
            }
        }

#if !__AVX2__
        pp = pp1;
        pp1 = pp + max_kk * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        __m128 _scales = _mm_load_ps((const float*)scales + i + ii);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m128i _w_shift = _mm_setzero_si128();
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                __m128 _p2 = _mm_load_ps(p0 + 8);
                __m128 _p3 = _mm_load_ps(p0 + 12);

                _p0 = _mm_mul_ps(_p0, _scales);
                _p1 = _mm_mul_ps(_p1, _scales);
                _p2 = _mm_mul_ps(_p2, _scales);
                _p3 = _mm_mul_ps(_p3, _scales);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);

                __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);
#if !__AVXVNNIINT8__
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 16;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm_storeu_si128((__m128i*)pp, _w_shift);
                pp += 16;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                _p0 = _mm_mul_ps(_p0, _scales);
                _p1 = _mm_mul_ps(_p1, _scales);
                __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p = _mm_load_ps(p0);
                _p = _mm_mul_ps(_p, _scales);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + A_hstep * 2);
                __m128 _p3 = _mm_loadu_ps(p0 + A_hstep * 3);

                _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);

                _p0 = _mm_mul_ps(_p0, _scales);
                _p1 = _mm_mul_ps(_p1, _scales);
                _p2 = _mm_mul_ps(_p2, _scales);
                _p3 = _mm_mul_ps(_p3, _scales);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);

                __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);
#if !__AVXVNNIINT8__
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm_storeu_si128((__m128i*)pp, _w_shift);
                pp += 16;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX2__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(A_hstep));

                __m128 _p0 = _mm_i32gather_ps(p0, _vindex, sizeof(float));
                __m128 _p1 = _mm_i32gather_ps(p0 + 1, _vindex, sizeof(float));
#else
                __m128 _p0 = _mm_setr_ps(p0[0], p0[A_hstep], p0[A_hstep * 2], p0[A_hstep * 3]);
                __m128 _p1 = _mm_setr_ps(p0[1], p0[A_hstep + 1], p0[A_hstep * 2 + 1], p0[A_hstep * 3 + 1]);
#endif
                _p0 = _mm_mul_ps(_p0, _scales);
                _p1 = _mm_mul_ps(_p1, _scales);
                __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
#if __AVX2__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(A_hstep));

                __m128 _p = _mm_i32gather_ps(p0, _vindex, sizeof(float));
#else
                __m128 _p = _mm_setr_ps(p0[0], p0[A_hstep], p0[A_hstep * 2], p0[A_hstep * 3]);
#endif
                _p = _mm_mul_ps(_p, _scales);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0++;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
#if __SSE2__
        __m128 _scales0 = _mm_set1_ps(scale0);
        __m128 _scales1 = _mm_set1_ps(scale1);
        __m128 _scales0011 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_scales0), _mm_castps_pd(_scales1)));
#endif // __SSE2__

        // if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            int w_shift0 = 0;
            int w_shift1 = 0;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep);
                _p0 = _mm_mul_ps(_p0, _scales0);
                _p1 = _mm_mul_ps(_p1, _scales1);
#if __AVX512VNNI__ || __AVXVNNI__
                int64_t v = float2int8_sse(_p0, _p1);
                *(int64_t*)pp = v;
#if !__AVXVNNIINT8__
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128 _t0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                __m128 _t1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 8;
                p0 += 4;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                pp += 8;
            }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _p1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                __m128 _p = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                _p = _mm_mul_ps(_p, _scales0011);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0 += 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale1);
                pp += 2;
                p0++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        const float scale = scales[i + ii];
#if __SSE2__
        __m128 _scale = _mm_set1_ps(scale);
#endif // __SSE2__

        // if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            int w_shift = 0;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p = _mm_loadu_ps(p0);
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp += 4;
                p0 += 4;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_compute_A_tile_fp32_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep * elempack : A.w * elempack;
    const int K = A.dims == 3 ? A.c : A.h;

    // NCNN_LOGE("transpose_compute_A_tile_int8_scales %d %d", max_ii, elempack);

    const float v127_B_scale = 127.f * B_scale;

    float* ps = (float*)scales + i;
    float* pods = (float*)out_descales + i;

    const int max_ii_unpacked = max_ii * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _v127_avx512 = _mm512_set1_ps(127.f);
    __m512 _v127_B_scale_avx512 = _mm512_set1_ps(v127_B_scale);
#endif // __AVX512F__
    __m256 _v127_avx = _mm256_set1_ps(127.f);
    __m256 _v127_B_scale_avx = _mm256_set1_ps(v127_B_scale);
#endif // __AVX__
    __m128 _v127 = _mm_set1_ps(127.f);
    __m128 _v127_B_scale = _mm_set1_ps(v127_B_scale);
#endif // __SSE2__

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 63 < max_ii_unpacked; ii += 64)
    {
        const float* ptr = (const float*)A + i * elempack + ii;

        __m512 _absmax0_avx512 = _mm512_setzero_ps();
        __m512 _absmax1_avx512 = _mm512_setzero_ps();
        __m512 _absmax2_avx512 = _mm512_setzero_ps();
        __m512 _absmax3_avx512 = _mm512_setzero_ps();

        for (int kk = 0; kk < K; kk++)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 32);
            __m512 _p3 = _mm512_loadu_ps(ptr + 48);
            _absmax0_avx512 = _mm512_max_ps(_absmax0_avx512, abs512_ps(_p0));
            _absmax1_avx512 = _mm512_max_ps(_absmax1_avx512, abs512_ps(_p1));
            _absmax2_avx512 = _mm512_max_ps(_absmax2_avx512, abs512_ps(_p2));
            _absmax3_avx512 = _mm512_max_ps(_absmax3_avx512, abs512_ps(_p3));
            ptr += A_hstep;
        }

        if (elempack == 16)
        {
            __m512 _tmp0 = _mm512_unpacklo_ps(_absmax0_avx512, _absmax2_avx512);
            __m512 _tmp1 = _mm512_unpackhi_ps(_absmax0_avx512, _absmax2_avx512);
            __m512 _tmp2 = _mm512_unpacklo_ps(_absmax1_avx512, _absmax3_avx512);
            __m512 _tmp3 = _mm512_unpackhi_ps(_absmax1_avx512, _absmax3_avx512);
            _absmax0_avx512 = _mm512_max_ps(_tmp0, _tmp1);
            _absmax1_avx512 = _mm512_max_ps(_tmp2, _tmp3);
            _tmp0 = _mm512_unpacklo_ps(_absmax0_avx512, _absmax1_avx512);
            _tmp1 = _mm512_unpackhi_ps(_absmax0_avx512, _absmax1_avx512);
            __m512 _absmax_avx512 = _mm512_max_ps(_tmp0, _tmp1);
            __m256 _absmax0_avx = _mm512_extractf32x8_ps(_absmax_avx512, 0);
            __m256 _absmax1_avx = _mm512_extractf32x8_ps(_absmax_avx512, 1);
            __m256 _absmax_avx = _mm256_max_ps(_absmax0_avx, _absmax1_avx);
            __m128 _absmax0 = _mm256_extractf128_ps(_absmax_avx, 0);
            __m128 _absmax1 = _mm256_extractf128_ps(_absmax_avx, 1);
            __m128 _absmax = _mm_max_ps(_absmax0, _absmax1);
            __m128 _scale0 = _mm_div_ps(_v127, _absmax);
            __m128 _out_descale0 = _mm_div_ps(_absmax, _v127_B_scale);
            _mm_store_ps(ps, _scale0);
            _mm_store_ps(pods, _out_descale0);
            ps += 4;
            pods += 4;
        }
        if (elempack == 8)
        {
            __m512 _tmp0 = _mm512_unpacklo_ps(_absmax0_avx512, _absmax2_avx512);
            __m512 _tmp1 = _mm512_unpackhi_ps(_absmax0_avx512, _absmax2_avx512);
            __m512 _tmp2 = _mm512_unpacklo_ps(_absmax1_avx512, _absmax3_avx512);
            __m512 _tmp3 = _mm512_unpackhi_ps(_absmax1_avx512, _absmax3_avx512);
            _absmax0_avx512 = _mm512_max_ps(_tmp0, _tmp1);
            _absmax1_avx512 = _mm512_max_ps(_tmp2, _tmp3);
            _tmp0 = _mm512_unpacklo_ps(_absmax0_avx512, _absmax1_avx512);
            _tmp1 = _mm512_unpackhi_ps(_absmax0_avx512, _absmax1_avx512);
            __m512 _absmax_avx512 = _mm512_max_ps(_tmp0, _tmp1);
            _absmax_avx512 = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15), _absmax_avx512);
            __m256 _absmax0_avx = _mm512_extractf32x8_ps(_absmax_avx512, 0);
            __m256 _absmax1_avx = _mm512_extractf32x8_ps(_absmax_avx512, 1);
            __m256 _absmax_avx = _mm256_max_ps(_absmax0_avx, _absmax1_avx);
            __m256 _scale = _mm256_div_ps(_v127_avx, _absmax_avx);
            __m256 _out_descale = _mm256_div_ps(_absmax_avx, _v127_B_scale_avx);
            _mm256_store_ps(ps, _scale);
            _mm256_store_ps(pods, _out_descale);
            ps += 8;
            pods += 8;
        }
        if (elempack == 4)
        {
            __m512 _tmp0 = _mm512_unpacklo_ps(_absmax0_avx512, _absmax2_avx512);
            __m512 _tmp1 = _mm512_unpackhi_ps(_absmax0_avx512, _absmax2_avx512);
            __m512 _tmp2 = _mm512_unpacklo_ps(_absmax1_avx512, _absmax3_avx512);
            __m512 _tmp3 = _mm512_unpackhi_ps(_absmax1_avx512, _absmax3_avx512);
            _absmax0_avx512 = _mm512_max_ps(_tmp0, _tmp1);
            _absmax1_avx512 = _mm512_max_ps(_tmp2, _tmp3);
            _tmp0 = _mm512_unpacklo_ps(_absmax0_avx512, _absmax1_avx512);
            _tmp1 = _mm512_unpackhi_ps(_absmax0_avx512, _absmax1_avx512);
            __m512 _absmax_avx512 = _mm512_max_ps(_tmp0, _tmp1);
            _absmax_avx512 = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), _absmax_avx512);
            __m512 _scale0 = _mm512_div_ps(_v127_avx512, _absmax_avx512);
            __m512 _out_descale0 = _mm512_div_ps(_absmax_avx512, _v127_B_scale_avx512);
            _mm512_store_ps(ps, _scale0);
            _mm512_store_ps(pods, _out_descale0);
            ps += 16;
            pods += 16;
        }
        if (elempack == 1)
        {
            __m512 _scale0 = _mm512_div_ps(_v127_avx512, _absmax0_avx512);
            __m512 _scale1 = _mm512_div_ps(_v127_avx512, _absmax1_avx512);
            __m512 _scale2 = _mm512_div_ps(_v127_avx512, _absmax2_avx512);
            __m512 _scale3 = _mm512_div_ps(_v127_avx512, _absmax3_avx512);
            __m512 _out_descale0 = _mm512_div_ps(_absmax0_avx512, _v127_B_scale_avx512);
            __m512 _out_descale1 = _mm512_div_ps(_absmax1_avx512, _v127_B_scale_avx512);
            __m512 _out_descale2 = _mm512_div_ps(_absmax2_avx512, _v127_B_scale_avx512);
            __m512 _out_descale3 = _mm512_div_ps(_absmax3_avx512, _v127_B_scale_avx512);
            _mm512_store_ps(ps, _scale0);
            _mm512_store_ps(ps + 16, _scale1);
            _mm512_store_ps(ps + 32, _scale2);
            _mm512_store_ps(ps + 48, _scale3);
            _mm512_store_ps(pods, _out_descale0);
            _mm512_store_ps(pods + 16, _out_descale1);
            _mm512_store_ps(pods + 32, _out_descale2);
            _mm512_store_ps(pods + 48, _out_descale3);
            ps += 64;
            pods += 64;
        }
    }
#endif // __AVX512F__
    for (; ii + 31 < max_ii_unpacked; ii += 32)
    {
        const float* ptr = (const float*)A + i * elempack + ii;

#if __AVX512F__
        __m512 _absmax0_avx512 = _mm512_setzero_ps();
        __m512 _absmax1_avx512 = _mm512_setzero_ps();
        __m512 _absmax2_avx512 = _mm512_setzero_ps();
        __m512 _absmax3_avx512 = _mm512_setzero_ps();
#else
        __m256 _absmax0_avx = _mm256_setzero_ps();
        __m256 _absmax1_avx = _mm256_setzero_ps();
        __m256 _absmax2_avx = _mm256_setzero_ps();
        __m256 _absmax3_avx = _mm256_setzero_ps();
#endif

        int kk = 0;
#if __AVX512F__
        for (; kk + 1 < K; kk += 2)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + A_hstep);
            __m512 _p3 = _mm512_loadu_ps(ptr + A_hstep + 16);
            _absmax0_avx512 = _mm512_max_ps(_absmax0_avx512, abs512_ps(_p0));
            _absmax1_avx512 = _mm512_max_ps(_absmax1_avx512, abs512_ps(_p1));
            _absmax2_avx512 = _mm512_max_ps(_absmax2_avx512, abs512_ps(_p2));
            _absmax3_avx512 = _mm512_max_ps(_absmax3_avx512, abs512_ps(_p3));
            ptr += A_hstep * 2;
        }
        _absmax0_avx512 = _mm512_max_ps(_absmax0_avx512, _absmax2_avx512);
        _absmax1_avx512 = _mm512_max_ps(_absmax1_avx512, _absmax3_avx512);
#endif // __AVX512F__
        for (; kk < K; kk++)
        {
#if __AVX512F__
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            _absmax0_avx512 = _mm512_max_ps(_absmax0_avx512, abs512_ps(_p0));
            _absmax1_avx512 = _mm512_max_ps(_absmax1_avx512, abs512_ps(_p1));
#else
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            __m256 _p2 = _mm256_loadu_ps(ptr + 16);
            __m256 _p3 = _mm256_loadu_ps(ptr + 24);
            _absmax0_avx = _mm256_max_ps(_absmax0_avx, abs256_ps(_p0));
            _absmax1_avx = _mm256_max_ps(_absmax1_avx, abs256_ps(_p1));
            _absmax2_avx = _mm256_max_ps(_absmax2_avx, abs256_ps(_p2));
            _absmax3_avx = _mm256_max_ps(_absmax3_avx, abs256_ps(_p3));
#endif
            ptr += A_hstep;
        }

#if __AVX512F__
        if (elempack == 16)
        {
            float absmax0 = _mm512_reduce_max_ps(_absmax0_avx512);
            float absmax1 = _mm512_reduce_max_ps(_absmax1_avx512);
            ps[0] = 127.f / absmax0;
            ps[1] = 127.f / absmax1;
            pods[0] = absmax0 / v127_B_scale;
            pods[1] = absmax1 / v127_B_scale;
            ps += 2;
            pods += 2;
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX512F__
            __m512 _tmp0 = _mm512_unpacklo_ps(_absmax0_avx512, _absmax1_avx512);
            __m512 _tmp1 = _mm512_unpackhi_ps(_absmax0_avx512, _absmax1_avx512);
            _tmp0 = _mm512_max_ps(_tmp0, _tmp1);
            __m256 _absmax0_avx = _mm512_extractf32x8_ps(_tmp0, 0);
            __m256 _absmax1_avx = _mm512_extractf32x8_ps(_tmp0, 1);
#else
            __m256 _tmp0 = _mm256_unpacklo_ps(_absmax0_avx, _absmax2_avx);
            __m256 _tmp1 = _mm256_unpackhi_ps(_absmax0_avx, _absmax2_avx);
            __m256 _tmp2 = _mm256_unpacklo_ps(_absmax1_avx, _absmax3_avx);
            __m256 _tmp3 = _mm256_unpackhi_ps(_absmax1_avx, _absmax3_avx);
            _absmax0_avx = _mm256_max_ps(_tmp0, _tmp1);
            _absmax1_avx = _mm256_max_ps(_tmp2, _tmp3);
#endif
            __m256 _t0 = _mm256_unpacklo_ps(_absmax0_avx, _absmax1_avx);
            __m256 _t1 = _mm256_unpackhi_ps(_absmax0_avx, _absmax1_avx);
            _t0 = _mm256_max_ps(_t0, _t1);
            __m128 _absmax0 = _mm256_extractf128_ps(_t0, 0);
            __m128 _absmax1 = _mm256_extractf128_ps(_t0, 1);
            __m128 _absmax = _mm_max_ps(_absmax0, _absmax1);
            __m128 _scale0 = _mm_div_ps(_v127, _absmax);
            __m128 _out_descale0 = _mm_div_ps(_absmax, _v127_B_scale);
            _mm_store_ps(ps, _scale0);
            _mm_store_ps(pods, _out_descale0);
            ps += 4;
            pods += 4;
        }
        if (elempack == 4)
        {
#if __AVX512F__
            __m512 _tmp0 = _mm512_unpacklo_ps(_absmax0_avx512, _absmax1_avx512);
            __m512 _tmp1 = _mm512_unpackhi_ps(_absmax0_avx512, _absmax1_avx512);
            _tmp0 = _mm512_max_ps(_tmp0, _tmp1);
            __m256 _absmax0_avx = _mm512_extractf32x8_ps(_tmp0, 0);
            __m256 _absmax1_avx = _mm512_extractf32x8_ps(_tmp0, 1);
#else
            __m256 _tmp0 = _mm256_unpacklo_ps(_absmax0_avx, _absmax2_avx);
            __m256 _tmp1 = _mm256_unpackhi_ps(_absmax0_avx, _absmax2_avx);
            __m256 _tmp2 = _mm256_unpacklo_ps(_absmax1_avx, _absmax3_avx);
            __m256 _tmp3 = _mm256_unpackhi_ps(_absmax1_avx, _absmax3_avx);
            _absmax0_avx = _mm256_max_ps(_tmp0, _tmp1);
            _absmax1_avx = _mm256_max_ps(_tmp2, _tmp3);
#endif
            __m256 _t0 = _mm256_unpacklo_ps(_absmax0_avx, _absmax1_avx);
            __m256 _t1 = _mm256_unpackhi_ps(_absmax0_avx, _absmax1_avx);
            __m256 _absmax_avx = _mm256_max_ps(_t0, _t1);
            __m128 _tt0 = _mm256_extractf128_ps(_absmax_avx, 0);
            __m128 _tt1 = _mm256_extractf128_ps(_absmax_avx, 1);
            __m128 _absmax0 = _mm_unpacklo_ps(_tt0, _tt1);
            __m128 _absmax1 = _mm_unpackhi_ps(_tt0, _tt1);
            _absmax_avx = combine4x2_ps(_absmax0, _absmax1);
            __m256 _scale = _mm256_div_ps(_v127_avx, _absmax_avx);
            __m256 _out_descale = _mm256_div_ps(_absmax_avx, _v127_B_scale_avx);
            _mm256_store_ps(ps, _scale);
            _mm256_store_ps(pods, _out_descale);
            ps += 8;
            pods += 8;
        }
        if (elempack == 1)
        {
#if __AVX512F__
            __m512 _scale0 = _mm512_div_ps(_v127_avx512, _absmax0_avx512);
            __m512 _scale1 = _mm512_div_ps(_v127_avx512, _absmax1_avx512);
            __m512 _out_descale0 = _mm512_div_ps(_absmax0_avx512, _v127_B_scale_avx512);
            __m512 _out_descale1 = _mm512_div_ps(_absmax1_avx512, _v127_B_scale_avx512);
            _mm512_store_ps(ps, _scale0);
            _mm512_store_ps(ps + 16, _scale1);
            _mm512_store_ps(pods, _out_descale0);
            _mm512_store_ps(pods + 16, _out_descale1);
#else
            __m256 _scale0 = _mm256_div_ps(_v127_avx, _absmax0_avx);
            __m256 _scale1 = _mm256_div_ps(_v127_avx, _absmax1_avx);
            __m256 _scale2 = _mm256_div_ps(_v127_avx, _absmax2_avx);
            __m256 _scale3 = _mm256_div_ps(_v127_avx, _absmax3_avx);
            __m256 _out_descale0 = _mm256_div_ps(_absmax0_avx, _v127_B_scale_avx);
            __m256 _out_descale1 = _mm256_div_ps(_absmax1_avx, _v127_B_scale_avx);
            __m256 _out_descale2 = _mm256_div_ps(_absmax2_avx, _v127_B_scale_avx);
            __m256 _out_descale3 = _mm256_div_ps(_absmax3_avx, _v127_B_scale_avx);
            _mm256_store_ps(ps, _scale0);
            _mm256_store_ps(ps + 8, _scale1);
            _mm256_store_ps(ps + 16, _scale2);
            _mm256_store_ps(ps + 24, _scale3);
            _mm256_store_ps(pods, _out_descale0);
            _mm256_store_ps(pods + 8, _out_descale1);
            _mm256_store_ps(pods + 16, _out_descale2);
            _mm256_store_ps(pods + 24, _out_descale3);
#endif
            ps += 32;
            pods += 32;
        }
    }
#endif // __AVX__
    for (; ii + 15 < max_ii_unpacked; ii += 16)
    {
        const float* ptr = (const float*)A + i * elempack + ii;

#if __AVX512F__
        __m512 _absmax_avx512 = _mm512_setzero_ps();
        __m512 _absmax1_avx512 = _mm512_setzero_ps();
        __m512 _absmax2_avx512 = _mm512_setzero_ps();
        __m512 _absmax3_avx512 = _mm512_setzero_ps();
#elif __AVX__
        __m256 _absmax0_avx = _mm256_setzero_ps();
        __m256 _absmax1_avx = _mm256_setzero_ps();
        __m256 _absmax2_avx = _mm256_setzero_ps();
        __m256 _absmax3_avx = _mm256_setzero_ps();
#else
        __m128 _absmax0 = _mm_setzero_ps();
        __m128 _absmax1 = _mm_setzero_ps();
        __m128 _absmax2 = _mm_setzero_ps();
        __m128 _absmax3 = _mm_setzero_ps();
#endif

        int kk = 0;
#if __AVX__
#if __AVX512F__
        for (; kk + 3 < K; kk += 4)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + A_hstep);
            __m512 _p2 = _mm512_loadu_ps(ptr + A_hstep * 2);
            __m512 _p3 = _mm512_loadu_ps(ptr + A_hstep * 3);
            _absmax_avx512 = _mm512_max_ps(_absmax_avx512, abs512_ps(_p0));
            _absmax1_avx512 = _mm512_max_ps(_absmax1_avx512, abs512_ps(_p1));
            _absmax2_avx512 = _mm512_max_ps(_absmax2_avx512, abs512_ps(_p2));
            _absmax3_avx512 = _mm512_max_ps(_absmax3_avx512, abs512_ps(_p3));
            ptr += A_hstep * 4;
        }
        _absmax_avx512 = _mm512_max_ps(_absmax_avx512, _absmax2_avx512);
        _absmax1_avx512 = _mm512_max_ps(_absmax1_avx512, _absmax3_avx512);
#endif // __AVX512F__
        for (; kk + 1 < K; kk += 2)
        {
#if __AVX512F__
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + A_hstep);
            _absmax_avx512 = _mm512_max_ps(_absmax_avx512, abs512_ps(_p0));
            _absmax1_avx512 = _mm512_max_ps(_absmax1_avx512, abs512_ps(_p1));
#else
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            __m256 _p2 = _mm256_loadu_ps(ptr + A_hstep);
            __m256 _p3 = _mm256_loadu_ps(ptr + A_hstep + 8);
            _absmax0_avx = _mm256_max_ps(_absmax0_avx, abs256_ps(_p0));
            _absmax1_avx = _mm256_max_ps(_absmax1_avx, abs256_ps(_p1));
            _absmax2_avx = _mm256_max_ps(_absmax2_avx, abs256_ps(_p2));
            _absmax3_avx = _mm256_max_ps(_absmax3_avx, abs256_ps(_p3));
#endif
            ptr += A_hstep * 2;
        }
#if __AVX512F__
        _absmax_avx512 = _mm512_max_ps(_absmax_avx512, _absmax1_avx512);
#else
        _absmax0_avx = _mm256_max_ps(_absmax0_avx, _absmax2_avx);
        _absmax1_avx = _mm256_max_ps(_absmax1_avx, _absmax3_avx);
#endif
#endif // __AVX__
        for (; kk < K; kk++)
        {
#if __AVX512F__
            __m512 _p = _mm512_loadu_ps(ptr);
            _absmax_avx512 = _mm512_max_ps(_absmax_avx512, abs512_ps(_p));
#elif __AVX__
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            _absmax0_avx = _mm256_max_ps(_absmax0_avx, abs256_ps(_p0));
            _absmax1_avx = _mm256_max_ps(_absmax1_avx, abs256_ps(_p1));
#else
            __m128 _p0 = _mm_loadu_ps(ptr);
            __m128 _p1 = _mm_loadu_ps(ptr + 4);
            __m128 _p2 = _mm_loadu_ps(ptr + 8);
            __m128 _p3 = _mm_loadu_ps(ptr + 12);
            _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p0));
            _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
            _absmax2 = _mm_max_ps(_absmax2, abs_ps(_p2));
            _absmax3 = _mm_max_ps(_absmax3, abs_ps(_p3));
#endif
            ptr += A_hstep;
        }

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            float absmax = _mm512_reduce_max_ps(_absmax_avx512);
            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX512F__
            __m256 _absmax0_avx = _mm512_extractf32x8_ps(_absmax_avx512, 0);
            __m256 _absmax1_avx = _mm512_extractf32x8_ps(_absmax_avx512, 1);
#endif
            float absmax0 = _mm256_reduce_max_ps(_absmax0_avx);
            float absmax1 = _mm256_reduce_max_ps(_absmax1_avx);
            ps[0] = 127.f / absmax0;
            ps[1] = 127.f / absmax1;
            pods[0] = absmax0 / v127_B_scale;
            pods[1] = absmax1 / v127_B_scale;
            ps += 2;
            pods += 2;
        }
#endif // __AVX__
        if (elempack == 4)
        {
#if __AVX__
#if __AVX512F__
            __m256 _absmax0_avx = _mm512_extractf32x8_ps(_absmax_avx512, 0);
            __m256 _absmax1_avx = _mm512_extractf32x8_ps(_absmax_avx512, 1);
#endif
            __m256 _tmp0 = _mm256_unpacklo_ps(_absmax0_avx, _absmax1_avx);
            __m256 _tmp1 = _mm256_unpackhi_ps(_absmax0_avx, _absmax1_avx);
            __m256 _absmax01_avx = _mm256_max_ps(_tmp0, _tmp1);
            __m128 _absmax0 = _mm256_extractf128_ps(_absmax01_avx, 0);
            __m128 _absmax1 = _mm256_extractf128_ps(_absmax01_avx, 1);
#else
            __m128 _tmp0 = _mm_unpacklo_ps(_absmax0, _absmax2);
            __m128 _tmp1 = _mm_unpackhi_ps(_absmax0, _absmax2);
            __m128 _tmp2 = _mm_unpacklo_ps(_absmax1, _absmax3);
            __m128 _tmp3 = _mm_unpackhi_ps(_absmax1, _absmax3);
            _absmax0 = _mm_max_ps(_tmp0, _tmp1);
            _absmax1 = _mm_max_ps(_tmp2, _tmp3);
#endif
            __m128 _t0 = _mm_unpacklo_ps(_absmax0, _absmax1);
            __m128 _t1 = _mm_unpackhi_ps(_absmax0, _absmax1);
            __m128 _absmax = _mm_max_ps(_t0, _t1);
            __m128 _scale0 = _mm_div_ps(_v127, _absmax);
            __m128 _out_descale0 = _mm_div_ps(_absmax, _v127_B_scale);
            _mm_store_ps(ps, _scale0);
            _mm_store_ps(pods, _out_descale0);
            ps += 4;
            pods += 4;
        }
        if (elempack == 1)
        {
#if __AVX512F__
            __m512 _scale = _mm512_div_ps(_v127_avx512, _absmax_avx512);
            __m512 _out_descale = _mm512_div_ps(_absmax_avx512, _v127_B_scale_avx512);
            _mm512_store_ps(ps, _scale);
            _mm512_store_ps(pods, _out_descale);
#elif __AVX__
            __m256 _scale0 = _mm256_div_ps(_v127_avx, _absmax0_avx);
            __m256 _scale1 = _mm256_div_ps(_v127_avx, _absmax1_avx);
            __m256 _out_descale0 = _mm256_div_ps(_absmax0_avx, _v127_B_scale_avx);
            __m256 _out_descale1 = _mm256_div_ps(_absmax1_avx, _v127_B_scale_avx);
            _mm256_store_ps(ps, _scale0);
            _mm256_store_ps(ps + 8, _scale1);
            _mm256_store_ps(pods, _out_descale0);
            _mm256_store_ps(pods + 8, _out_descale1);
#else
            __m128 _scale0 = _mm_div_ps(_v127, _absmax0);
            __m128 _scale1 = _mm_div_ps(_v127, _absmax1);
            __m128 _scale2 = _mm_div_ps(_v127, _absmax2);
            __m128 _scale3 = _mm_div_ps(_v127, _absmax3);
            __m128 _out_descale0 = _mm_div_ps(_absmax0, _v127_B_scale);
            __m128 _out_descale1 = _mm_div_ps(_absmax1, _v127_B_scale);
            __m128 _out_descale2 = _mm_div_ps(_absmax2, _v127_B_scale);
            __m128 _out_descale3 = _mm_div_ps(_absmax3, _v127_B_scale);
            _mm_store_ps(ps, _scale0);
            _mm_store_ps(ps + 4, _scale1);
            _mm_store_ps(ps + 8, _scale2);
            _mm_store_ps(ps + 12, _scale3);
            _mm_store_ps(pods, _out_descale0);
            _mm_store_ps(pods + 4, _out_descale1);
            _mm_store_ps(pods + 8, _out_descale2);
            _mm_store_ps(pods + 12, _out_descale3);
#endif
            ps += 16;
            pods += 16;
        }
    }
    for (; ii + 7 < max_ii_unpacked; ii += 8)
    {
        const float* ptr = (const float*)A + i * elempack + ii;

#if __AVX__
        __m256 _absmax_avx = _mm256_setzero_ps();
        __m256 _absmax1_avx = _mm256_setzero_ps();
        __m256 _absmax2_avx = _mm256_setzero_ps();
        __m256 _absmax3_avx = _mm256_setzero_ps();
#else
        __m128 _absmax0 = _mm_setzero_ps();
        __m128 _absmax1 = _mm_setzero_ps();
        __m128 _absmax2 = _mm_setzero_ps();
        __m128 _absmax3 = _mm_setzero_ps();
#endif

        int kk = 0;
#if __AVX__
        for (; kk + 3 < K; kk += 4)
        {
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + A_hstep);
            __m256 _p2 = _mm256_loadu_ps(ptr + A_hstep * 2);
            __m256 _p3 = _mm256_loadu_ps(ptr + A_hstep * 3);
            _absmax_avx = _mm256_max_ps(_absmax_avx, abs256_ps(_p0));
            _absmax1_avx = _mm256_max_ps(_absmax1_avx, abs256_ps(_p1));
            _absmax2_avx = _mm256_max_ps(_absmax2_avx, abs256_ps(_p2));
            _absmax3_avx = _mm256_max_ps(_absmax3_avx, abs256_ps(_p3));
            ptr += A_hstep * 4;
        }
        _absmax_avx = _mm256_max_ps(_absmax_avx, _absmax2_avx);
        _absmax1_avx = _mm256_max_ps(_absmax1_avx, _absmax3_avx);
#endif // __AVX__
        for (; kk + 1 < K; kk += 2)
        {
#if __AVX__
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + A_hstep);
            _absmax_avx = _mm256_max_ps(_absmax_avx, abs256_ps(_p0));
            _absmax1_avx = _mm256_max_ps(_absmax1_avx, abs256_ps(_p1));
#else
            __m128 _p0 = _mm_loadu_ps(ptr);
            __m128 _p1 = _mm_loadu_ps(ptr + 4);
            __m128 _p2 = _mm_loadu_ps(ptr + A_hstep);
            __m128 _p3 = _mm_loadu_ps(ptr + A_hstep + 4);
            _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p0));
            _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
            _absmax2 = _mm_max_ps(_absmax2, abs_ps(_p2));
            _absmax3 = _mm_max_ps(_absmax3, abs_ps(_p3));
#endif
            ptr += A_hstep * 2;
        }
#if __AVX__
        _absmax_avx = _mm256_max_ps(_absmax_avx, _absmax1_avx);
#else
        _absmax0 = _mm_max_ps(_absmax0, _absmax2);
        _absmax1 = _mm_max_ps(_absmax1, _absmax3);
#endif
        for (; kk < K; kk++)
        {
#if __AVX__
            __m256 _p = _mm256_loadu_ps(ptr);
            _absmax_avx = _mm256_max_ps(_absmax_avx, abs256_ps(_p));
#else
            __m128 _p0 = _mm_loadu_ps(ptr);
            __m128 _p1 = _mm_loadu_ps(ptr + 4);
            _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p0));
            _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
#endif
            ptr += A_hstep;
        }

#if __AVX__
        if (elempack == 8)
        {
            float absmax = _mm256_reduce_max_ps(_absmax_avx);
            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
#endif // __AVX__
        if (elempack == 4)
        {
#if __AVX__
            __m128 _absmax0 = _mm256_extractf128_ps(_absmax_avx, 0);
            __m128 _absmax1 = _mm256_extractf128_ps(_absmax_avx, 1);
#endif
            float absmax0 = _mm_reduce_max_ps(_absmax0);
            float absmax1 = _mm_reduce_max_ps(_absmax1);
            ps[0] = 127.f / absmax0;
            ps[1] = 127.f / absmax1;
            pods[0] = absmax0 / v127_B_scale;
            pods[1] = absmax1 / v127_B_scale;
            ps += 2;
            pods += 2;
        }
        if (elempack == 1)
        {
#if __AVX__
            __m256 _scale = _mm256_div_ps(_v127_avx, _absmax_avx);
            __m256 _out_descale = _mm256_div_ps(_absmax_avx, _v127_B_scale_avx);
            _mm256_store_ps(ps, _scale);
            _mm256_store_ps(pods, _out_descale);
#else
            __m128 _scale0 = _mm_div_ps(_v127, _absmax0);
            __m128 _scale1 = _mm_div_ps(_v127, _absmax1);
            __m128 _out_descale0 = _mm_div_ps(_absmax0, _v127_B_scale);
            __m128 _out_descale1 = _mm_div_ps(_absmax1, _v127_B_scale);
            _mm_store_ps(ps, _scale0);
            _mm_store_ps(ps + 4, _scale1);
            _mm_store_ps(pods, _out_descale0);
            _mm_store_ps(pods + 4, _out_descale1);
#endif
            ps += 8;
            pods += 8;
        }
    }
#endif // __SSE2__
    for (; ii + 3 < max_ii_unpacked; ii += 4)
    {
        const float* ptr = (const float*)A + i * elempack + ii;

#if __SSE2__
        __m128 _absmax = _mm_setzero_ps();
        __m128 _absmax1 = _mm_setzero_ps();
        __m128 _absmax2 = _mm_setzero_ps();
        __m128 _absmax3 = _mm_setzero_ps();
#else
        float absmax0 = 0.f;
        float absmax1 = 0.f;
        float absmax2 = 0.f;
        float absmax3 = 0.f;
#endif

        int kk = 0;
#if __SSE2__
        for (; kk + 3 < K; kk += 4)
        {
            __m128 _p0 = _mm_loadu_ps(ptr);
            __m128 _p1 = _mm_loadu_ps(ptr + A_hstep);
            __m128 _p2 = _mm_loadu_ps(ptr + A_hstep * 2);
            __m128 _p3 = _mm_loadu_ps(ptr + A_hstep * 3);
            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
            _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
            _absmax2 = _mm_max_ps(_absmax2, abs_ps(_p2));
            _absmax3 = _mm_max_ps(_absmax3, abs_ps(_p3));
            ptr += A_hstep * 4;
        }
        _absmax = _mm_max_ps(_absmax, _absmax2);
        _absmax1 = _mm_max_ps(_absmax1, _absmax3);
        for (; kk + 1 < K; kk += 2)
        {
            __m128 _p0 = _mm_loadu_ps(ptr);
            __m128 _p1 = _mm_loadu_ps(ptr + A_hstep);
            _absmax = _mm_max_ps(_absmax, abs_ps(_p0));
            _absmax1 = _mm_max_ps(_absmax1, abs_ps(_p1));
            ptr += A_hstep * 2;
        }
        _absmax = _mm_max_ps(_absmax, _absmax1);
#endif // __SSE2__
        for (; kk < K; kk++)
        {
#if __SSE2__
            __m128 _p = _mm_loadu_ps(ptr);
            _absmax = _mm_max_ps(_absmax, abs_ps(_p));
#else
            absmax0 = std::max(absmax0, (float)fabsf(ptr[0]));
            absmax1 = std::max(absmax1, (float)fabsf(ptr[1]));
            absmax2 = std::max(absmax2, (float)fabsf(ptr[2]));
            absmax3 = std::max(absmax3, (float)fabsf(ptr[3]));
#endif
            ptr += A_hstep;
        }

#if __SSE2__
        if (elempack == 4)
        {
            float absmax = _mm_reduce_max_ps(_absmax);
            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
#endif // __SSE2__
        if (elempack == 1)
        {
#if __SSE2__
            __m128 _scale = _mm_div_ps(_v127, _absmax);
            __m128 _out_descale = _mm_div_ps(_absmax, _v127_B_scale);
            _mm_store_ps(ps, _scale);
            _mm_store_ps(pods, _out_descale);
#else
            ps[0] = 127.f / absmax0;
            ps[1] = 127.f / absmax1;
            ps[2] = 127.f / absmax2;
            ps[3] = 127.f / absmax3;
            pods[0] = absmax0 / v127_B_scale;
            pods[1] = absmax1 / v127_B_scale;
            pods[2] = absmax2 / v127_B_scale;
            pods[3] = absmax3 / v127_B_scale;
#endif
            ps += 4;
            pods += 4;
        }
    }
    for (; ii + 1 < max_ii_unpacked; ii += 2)
    {
        const float* ptr = (const float*)A + i * elempack + ii;

        float absmax0 = 0.f;
        float absmax1 = 0.f;

        int kk = 0;
#if __AVX512F__
        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(A_hstep));

        __m512 _absmax0_avx512 = _mm512_setzero_ps();
        __m512 _absmax1_avx512 = _mm512_setzero_ps();
        for (; kk + 15 < K; kk += 16)
        {
            __m512 _p0 = _mm512_i32gather_ps(_vindex, ptr, sizeof(float));
            __m512 _p1 = _mm512_i32gather_ps(_vindex, ptr + 1, sizeof(float));
            _absmax0_avx512 = _mm512_max_ps(_absmax0_avx512, abs512_ps(_p0));
            _absmax1_avx512 = _mm512_max_ps(_absmax1_avx512, abs512_ps(_p1));
            ptr += A_hstep * 16;
        }
        absmax0 = _mm512_comp_reduce_max_ps(_absmax0_avx512);
        absmax1 = _mm512_comp_reduce_max_ps(_absmax1_avx512);
#endif // __AVX512F__
        for (; kk < K; kk++)
        {
            absmax0 = std::max(absmax0, (float)fabsf(ptr[0]));
            absmax1 = std::max(absmax1, (float)fabsf(ptr[1]));
            ptr += A_hstep;
        }

        ps[0] = 127.f / absmax0;
        ps[1] = 127.f / absmax1;
        pods[0] = absmax0 / v127_B_scale;
        pods[1] = absmax1 / v127_B_scale;
        ps += 2;
        pods += 2;
    }
    for (; ii < max_ii_unpacked; ii++)
    {
        const float* ptr = (const float*)A + i * elempack + ii;

        float absmax = 0.f;

        int kk = 0;
#if __AVX512F__
        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(A_hstep));

        __m512 _absmax_avx512 = _mm512_setzero_ps();
        for (; kk + 15 < K; kk += 16)
        {
            __m512 _p = _mm512_i32gather_ps(_vindex, ptr, sizeof(float));
            _absmax_avx512 = _mm512_max_ps(_absmax_avx512, abs512_ps(_p));
            ptr += A_hstep * 16;
        }
        absmax = _mm512_comp_reduce_max_ps(_absmax_avx512);
#endif // __AVX512F__
        for (; kk < K; kk++)
        {
            absmax = std::max(absmax, (float)fabsf(ptr[0]));
            ptr += A_hstep;
        }

        ps[0] = 127.f / absmax;
        pods[0] = absmax / v127_B_scale;
        ps++;
        pods++;
    }
}

static void transpose_pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_pack_A_tile_fp32_to_int8_avx512vnni(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        transpose_pack_A_tile_fp32_to_int8_avxvnniint8(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_pack_A_tile_fp32_to_int8_avxvnni(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_pack_A_tile_fp32_to_int8_avx2(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("transpose_pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    signed char* pp = (signed char*)AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        __m512 _scales = _mm512_load_ps((const float*)scales + i + ii);
#if __AVX512VNNI__
        __m512i _w_shift = _mm512_setzero_si512();
        __m512i _v127 = _mm512_set1_epi8(127);
#endif

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);
                __m512 _p4 = _mm512_load_ps(p0 + 64);
                __m512 _p5 = _mm512_load_ps(p0 + 80);
                __m512 _p6 = _mm512_load_ps(p0 + 96);
                __m512 _p7 = _mm512_load_ps(p0 + 112);
                __m512 _p8 = _mm512_load_ps(p0 + 128);
                __m512 _p9 = _mm512_load_ps(p0 + 128 + 16);
                __m512 _pa = _mm512_load_ps(p0 + 128 + 32);
                __m512 _pb = _mm512_load_ps(p0 + 128 + 48);
                __m512 _pc = _mm512_load_ps(p0 + 128 + 64);
                __m512 _pd = _mm512_load_ps(p0 + 128 + 80);
                __m512 _pe = _mm512_load_ps(p0 + 128 + 96);
                __m512 _pf = _mm512_load_ps(p0 + 128 + 112);

                _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(scales[i + ii]));
                _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(scales[i + ii + 1]));
                _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(scales[i + ii + 2]));
                _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(scales[i + ii + 3]));
                _p4 = _mm512_mul_ps(_p4, _mm512_set1_ps(scales[i + ii + 4]));
                _p5 = _mm512_mul_ps(_p5, _mm512_set1_ps(scales[i + ii + 5]));
                _p6 = _mm512_mul_ps(_p6, _mm512_set1_ps(scales[i + ii + 6]));
                _p7 = _mm512_mul_ps(_p7, _mm512_set1_ps(scales[i + ii + 7]));
                _p8 = _mm512_mul_ps(_p8, _mm512_set1_ps(scales[i + ii + 8]));
                _p9 = _mm512_mul_ps(_p9, _mm512_set1_ps(scales[i + ii + 9]));
                _pa = _mm512_mul_ps(_pa, _mm512_set1_ps(scales[i + ii + 10]));
                _pb = _mm512_mul_ps(_pb, _mm512_set1_ps(scales[i + ii + 11]));
                _pc = _mm512_mul_ps(_pc, _mm512_set1_ps(scales[i + ii + 12]));
                _pd = _mm512_mul_ps(_pd, _mm512_set1_ps(scales[i + ii + 13]));
                _pe = _mm512_mul_ps(_pe, _mm512_set1_ps(scales[i + ii + 14]));
                _pf = _mm512_mul_ps(_pf, _mm512_set1_ps(scales[i + ii + 15]));

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);
                __m128i _pp8 = float2int8_avx512(_p8);
                __m128i _pp9 = float2int8_avx512(_p9);
                __m128i _ppa = float2int8_avx512(_pa);
                __m128i _ppb = float2int8_avx512(_pb);
                __m128i _ppc = float2int8_avx512(_pc);
                __m128i _ppd = float2int8_avx512(_pd);
                __m128i _ppe = float2int8_avx512(_pe);
                __m128i _ppf = float2int8_avx512(_pf);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp4, _pp8, _ppc);
                __m512i _t1 = combine4x4_epi32(_pp1, _pp5, _pp9, _ppd);
                __m512i _t2 = combine4x4_epi32(_pp2, _pp6, _ppa, _ppe);
                __m512i _t3 = combine4x4_epi32(_pp3, _pp7, _ppb, _ppf);

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

                _mm512_store_si512((__m512i*)pp, _t0);
                _mm512_store_si512((__m512i*)(pp + 64), _t1);
                _mm512_store_si512((__m512i*)(pp + 128), _t2);
                _mm512_store_si512((__m512i*)(pp + 192), _t3);

                pp += 256;
                p0 += A_hstep * 16;
            }
            if (max_kk >= 4)
            {
                _mm512_store_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);
                __m512 _p4 = _mm512_load_ps(p0 + 64);
                __m512 _p5 = _mm512_load_ps(p0 + 80);
                __m512 _p6 = _mm512_load_ps(p0 + 96);
                __m512 _p7 = _mm512_load_ps(p0 + 112);
                __m512 _p8 = _mm512_load_ps(p0 + 128);
                __m512 _p9 = _mm512_load_ps(p0 + 128 + 16);
                __m512 _pa = _mm512_load_ps(p0 + 128 + 32);
                __m512 _pb = _mm512_load_ps(p0 + 128 + 48);
                __m512 _pc = _mm512_load_ps(p0 + 128 + 64);
                __m512 _pd = _mm512_load_ps(p0 + 128 + 80);
                __m512 _pe = _mm512_load_ps(p0 + 128 + 96);
                __m512 _pf = _mm512_load_ps(p0 + 128 + 112);

                _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(scales[i + ii]));
                _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(scales[i + ii + 1]));
                _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(scales[i + ii + 2]));
                _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(scales[i + ii + 3]));
                _p4 = _mm512_mul_ps(_p4, _mm512_set1_ps(scales[i + ii + 4]));
                _p5 = _mm512_mul_ps(_p5, _mm512_set1_ps(scales[i + ii + 5]));
                _p6 = _mm512_mul_ps(_p6, _mm512_set1_ps(scales[i + ii + 6]));
                _p7 = _mm512_mul_ps(_p7, _mm512_set1_ps(scales[i + ii + 7]));
                _p8 = _mm512_mul_ps(_p8, _mm512_set1_ps(scales[i + ii + 8]));
                _p9 = _mm512_mul_ps(_p9, _mm512_set1_ps(scales[i + ii + 9]));
                _pa = _mm512_mul_ps(_pa, _mm512_set1_ps(scales[i + ii + 10]));
                _pb = _mm512_mul_ps(_pb, _mm512_set1_ps(scales[i + ii + 11]));
                _pc = _mm512_mul_ps(_pc, _mm512_set1_ps(scales[i + ii + 12]));
                _pd = _mm512_mul_ps(_pd, _mm512_set1_ps(scales[i + ii + 13]));
                _pe = _mm512_mul_ps(_pe, _mm512_set1_ps(scales[i + ii + 14]));
                _pf = _mm512_mul_ps(_pf, _mm512_set1_ps(scales[i + ii + 15]));

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);
                __m128i _pp8 = float2int8_avx512(_p8);
                __m128i _pp9 = float2int8_avx512(_p9);
                __m128i _ppa = float2int8_avx512(_pa);
                __m128i _ppb = float2int8_avx512(_pb);
                __m128i _ppc = float2int8_avx512(_pc);
                __m128i _ppd = float2int8_avx512(_pd);
                __m128i _ppe = float2int8_avx512(_pe);
                __m128i _ppf = float2int8_avx512(_pf);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp4, _pp8, _ppc);
                __m512i _t1 = combine4x4_epi32(_pp1, _pp5, _pp9, _ppd);
                __m512i _t2 = combine4x4_epi32(_pp2, _pp6, _ppa, _ppe);
                __m512i _t3 = combine4x4_epi32(_pp3, _pp7, _ppb, _ppf);

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

                _mm512_store_si512((__m512i*)pp, _t0);
                _mm512_store_si512((__m512i*)(pp + 64), _t1);
                _mm512_store_si512((__m512i*)(pp + 128), _t2);
                _mm512_store_si512((__m512i*)(pp + 192), _t3);

                pp += 256;
                p0 += A_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 8)
        {
            __m512 _scales0 = _scales;
            __m512 _scales1 = _scales;
            __m512 _scales2 = _scales;
            __m512 _scales3 = _scales;
            __m512 _scales4 = _scales;
            __m512 _scales5 = _scales;
            __m512 _scales6 = _scales;
            __m512 _scales7 = _scales;
            transpose16x8_ps(_scales0, _scales1, _scales2, _scales3, _scales4, _scales5, _scales6, _scales7);

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + 32);
                __m512 _p3 = _mm512_loadu_ps(p0 + 48);
                __m512 _p4 = _mm512_loadu_ps(p0 + 64);
                __m512 _p5 = _mm512_loadu_ps(p0 + 80);
                __m512 _p6 = _mm512_loadu_ps(p0 + 96);
                __m512 _p7 = _mm512_loadu_ps(p0 + 112);

                _p0 = _mm512_mul_ps(_p0, _scales0);
                _p1 = _mm512_mul_ps(_p1, _scales1);
                _p2 = _mm512_mul_ps(_p2, _scales2);
                _p3 = _mm512_mul_ps(_p3, _scales3);
                _p4 = _mm512_mul_ps(_p4, _scales4);
                _p5 = _mm512_mul_ps(_p5, _scales5);
                _p6 = _mm512_mul_ps(_p6, _scales6);
                _p7 = _mm512_mul_ps(_p7, _scales7);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp2, _pp4, _pp6);
                __m512i _t1 = combine4x4_epi32(_pp1, _pp3, _pp5, _pp7);

                __m512i _t2 = _mm512_unpacklo_epi32(_t0, _t1);
                __m512i _t3 = _mm512_unpackhi_epi32(_t0, _t1);
                __m512i _ppa = _mm512_unpacklo_epi32(_t2, _t3);
                __m512i _ppb = _mm512_unpackhi_epi32(_t2, _t3);

                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _ppa);
                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _ppb);

                _mm512_store_si512((__m512i*)pp, _ppa);
                _mm512_store_si512((__m512i*)(pp + 64), _ppb);

                pp += 128;
                p0 += A_hstep * 8;
            }
            if (max_kk >= 4)
            {
                _mm512_store_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#else  // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + 32);
                __m512 _p3 = _mm512_loadu_ps(p0 + 48);
                __m512 _p4 = _mm512_loadu_ps(p0 + 64);
                __m512 _p5 = _mm512_loadu_ps(p0 + 80);
                __m512 _p6 = _mm512_loadu_ps(p0 + 96);
                __m512 _p7 = _mm512_loadu_ps(p0 + 112);

                _p0 = _mm512_mul_ps(_p0, _scales0);
                _p1 = _mm512_mul_ps(_p1, _scales1);
                _p2 = _mm512_mul_ps(_p2, _scales2);
                _p3 = _mm512_mul_ps(_p3, _scales3);
                _p4 = _mm512_mul_ps(_p4, _scales4);
                _p5 = _mm512_mul_ps(_p5, _scales5);
                _p6 = _mm512_mul_ps(_p6, _scales6);
                _p7 = _mm512_mul_ps(_p7, _scales7);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp2, _pp4, _pp6);
                __m512i _t1 = combine4x4_epi32(_pp1, _pp3, _pp5, _pp7);

                __m512i _t2 = _mm512_unpacklo_epi16(_t0, _t1);
                __m512i _t3 = _mm512_unpackhi_epi16(_t0, _t1);
                _t0 = _mm512_unpacklo_epi16(_t2, _t3);
                _t1 = _mm512_unpackhi_epi16(_t2, _t3);
                _t0 = _mm512_permutex_epi64(_t0, _MM_SHUFFLE(3, 1, 2, 0));
                _t1 = _mm512_permutex_epi64(_t1, _MM_SHUFFLE(3, 1, 2, 0));
                __m512i _ppa = _mm512_shuffle_i32x4(_t0, _t0, _MM_SHUFFLE(3, 1, 2, 0));
                __m512i _ppb = _mm512_shuffle_i32x4(_t1, _t1, _MM_SHUFFLE(3, 1, 2, 0));

                _mm512_store_si512((__m512i*)pp, _ppa);
                _mm512_store_si512((__m512i*)(pp + 64), _ppb);

                pp += 128;
                p0 += A_hstep * 8;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 4)
        {
            __m512 _scales0 = _scales;
            __m512 _scales1 = _scales;
            __m512 _scales2 = _scales;
            __m512 _scales3 = _scales;
            transpose16x4_ps(_scales0, _scales1, _scales2, _scales3);

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + 32);
                __m512 _p3 = _mm512_loadu_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scales0);
                _p1 = _mm512_mul_ps(_p1, _scales1);
                _p2 = _mm512_mul_ps(_p2, _scales2);
                _p3 = _mm512_mul_ps(_p3, _scales3);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _pp);

                _mm512_store_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                _mm512_store_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#else  // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + 32);
                __m512 _p3 = _mm512_loadu_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scales0);
                _p1 = _mm512_mul_ps(_p1, _scales1);
                _p2 = _mm512_mul_ps(_p2, _scales2);
                _p3 = _mm512_mul_ps(_p3, _scales3);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                __m256i _pp02 = combine4x2_epi32(_pp0, _pp2);
                __m256i _pp13 = combine4x2_epi32(_pp1, _pp3);

                __m256i _t0 = _mm256_unpacklo_epi16(_pp02, _pp13);
                __m256i _t1 = _mm256_unpackhi_epi16(_pp02, _pp13);
                __m256i _t2 = _mm256_unpacklo_epi16(_t0, _t1);
                __m256i _t3 = _mm256_unpackhi_epi16(_t0, _t1);
                _t0 = _mm256_unpacklo_epi16(_t2, _t3);
                _t1 = _mm256_unpackhi_epi16(_t2, _t3);

                _mm256_store_si256((__m256i*)pp, _t0);
                _mm256_store_si256((__m256i*)(pp + 32), _t1);

                pp += 64;
                p0 += A_hstep * 4;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + A_hstep);
                __m512 _p2 = _mm512_loadu_ps(p0 + A_hstep * 2);
                __m512 _p3 = _mm512_loadu_ps(p0 + A_hstep * 3);

                _p0 = _mm512_mul_ps(_p0, _scales);
                _p1 = _mm512_mul_ps(_p1, _scales);
                _p2 = _mm512_mul_ps(_p2, _scales);
                _p3 = _mm512_mul_ps(_p3, _scales);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose16x4_epi8(_pp0, _pp1, _pp2, _pp3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _pp);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                _mm512_storeu_si512((__m512i*)pp, _w_shift);
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + A_hstep);

                _p0 = _mm512_mul_ps(_p0, _scales);
                _p1 = _mm512_mul_ps(_p1, _scales);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

                // transpose16x2_epi8
                __m128i _t0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                __m512 _p = _mm512_loadu_ps(p0);

                _p = _mm512_mul_ps(_p, _scales);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += A_hstep;
            }
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    signed char* pp1 = pp + max_kk * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        __m256 _scales = _mm256_load_ps((const float*)scales + i + ii);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m256i _w_shift = _mm256_setzero_si256();
        __m256i _v127 = _mm256_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            __m512i _w_shift_avx512 = _mm512_setzero_si512();
            __m512i _v127_avx512 = _mm512_set1_epi8(127);
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);
                __m512 _p4 = _mm512_load_ps(p0 + 64);
                __m512 _p5 = _mm512_load_ps(p0 + 80);
                __m512 _p6 = _mm512_load_ps(p0 + 96);
                __m512 _p7 = _mm512_load_ps(p0 + 112);

                _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(scales[i + ii]));
                _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(scales[i + ii + 1]));
                _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(scales[i + ii + 2]));
                _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(scales[i + ii + 3]));
                _p4 = _mm512_mul_ps(_p4, _mm512_set1_ps(scales[i + ii + 4]));
                _p5 = _mm512_mul_ps(_p5, _mm512_set1_ps(scales[i + ii + 5]));
                _p6 = _mm512_mul_ps(_p6, _mm512_set1_ps(scales[i + ii + 6]));
                _p7 = _mm512_mul_ps(_p7, _mm512_set1_ps(scales[i + ii + 7]));

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);

                transpose4x8_epi32(_pp0, _pp1, _pp2, _pp3, _pp4, _pp5, _pp6, _pp7);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);
                __m512i _t1 = combine4x4_epi32(_pp4, _pp5, _pp6, _pp7);

                _w_shift_avx512 = _mm512_dpbusd_epi32(_w_shift_avx512, _v127_avx512, _t0);
                _w_shift_avx512 = _mm512_dpbusd_epi32(_w_shift_avx512, _v127_avx512, _t1);

                _mm512_store_si512((__m512i*)pp, _t0);
                _mm512_store_si512((__m512i*)(pp + 64), _t1);

                pp += 128;
                p0 += A_hstep * 16;
            }
            if (max_kk >= 4)
            {
                _w_shift = _mm256_add_epi32(_mm512_extracti32x8_epi32(_w_shift_avx512, 0), _mm512_extracti32x8_epi32(_w_shift_avx512, 1));
                _mm256_store_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);
                __m512 _p4 = _mm512_load_ps(p0 + 64);
                __m512 _p5 = _mm512_load_ps(p0 + 80);
                __m512 _p6 = _mm512_load_ps(p0 + 96);
                __m512 _p7 = _mm512_load_ps(p0 + 112);

                _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(scales[i + ii]));
                _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(scales[i + ii + 1]));
                _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(scales[i + ii + 2]));
                _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(scales[i + ii + 3]));
                _p4 = _mm512_mul_ps(_p4, _mm512_set1_ps(scales[i + ii + 4]));
                _p5 = _mm512_mul_ps(_p5, _mm512_set1_ps(scales[i + ii + 5]));
                _p6 = _mm512_mul_ps(_p6, _mm512_set1_ps(scales[i + ii + 6]));
                _p7 = _mm512_mul_ps(_p7, _mm512_set1_ps(scales[i + ii + 7]));

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);

                transpose8x8_epi16(_pp0, _pp1, _pp2, _pp3, _pp4, _pp5, _pp6, _pp7);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);
                __m512i _t1 = combine4x4_epi32(_pp4, _pp5, _pp6, _pp7);

                _mm512_store_si512((__m512i*)pp, _t0);
                _mm512_store_si512((__m512i*)(pp + 64), _t1);

                pp += 128;
                p0 += A_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);
                __m256 _p4 = _mm256_load_ps(p0 + 32);
                __m256 _p5 = _mm256_load_ps(p0 + 40);
                __m256 _p6 = _mm256_load_ps(p0 + 48);
                __m256 _p7 = _mm256_load_ps(p0 + 56);

                _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(scales[i + ii]));
                _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(scales[i + ii + 1]));
                _p2 = _mm256_mul_ps(_p2, _mm256_set1_ps(scales[i + ii + 2]));
                _p3 = _mm256_mul_ps(_p3, _mm256_set1_ps(scales[i + ii + 3]));
                _p4 = _mm256_mul_ps(_p4, _mm256_set1_ps(scales[i + ii + 4]));
                _p5 = _mm256_mul_ps(_p5, _mm256_set1_ps(scales[i + ii + 5]));
                _p6 = _mm256_mul_ps(_p6, _mm256_set1_ps(scales[i + ii + 6]));
                _p7 = _mm256_mul_ps(_p7, _mm256_set1_ps(scales[i + ii + 7]));

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);
                __m128i _pp2 = float2int8_avx(_p4, _p6);
                __m128i _pp3 = float2int8_avx(_p5, _p7);

                __m256i _t0 = combine4x2_epi32(_pp0, _pp2);
                __m256i _t1 = combine4x2_epi32(_pp1, _pp3);

                __m256i _t2 = _mm256_unpacklo_epi32(_t0, _t1);
                __m256i _t3 = _mm256_unpackhi_epi32(_t0, _t1);
                _t0 = _mm256_unpacklo_epi64(_t2, _t3);
                _t1 = _mm256_unpackhi_epi64(_t2, _t3);
#if !__AVXVNNIINT8__
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _t0);
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _t1);
#endif // !__AVXVNNIINT8__
                _mm256_store_si256((__m256i*)pp, _t0);
                _mm256_store_si256((__m256i*)(pp + 32), _t1);

                pp += 64;
                p0 += A_hstep * 8;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm256_store_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);
                __m256 _p4 = _mm256_load_ps(p0 + 32);
                __m256 _p5 = _mm256_load_ps(p0 + 40);
                __m256 _p6 = _mm256_load_ps(p0 + 48);
                __m256 _p7 = _mm256_load_ps(p0 + 56);

                _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(scales[i + ii]));
                _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(scales[i + ii + 1]));
                _p2 = _mm256_mul_ps(_p2, _mm256_set1_ps(scales[i + ii + 2]));
                _p3 = _mm256_mul_ps(_p3, _mm256_set1_ps(scales[i + ii + 3]));
                _p4 = _mm256_mul_ps(_p4, _mm256_set1_ps(scales[i + ii + 4]));
                _p5 = _mm256_mul_ps(_p5, _mm256_set1_ps(scales[i + ii + 5]));
                _p6 = _mm256_mul_ps(_p6, _mm256_set1_ps(scales[i + ii + 6]));
                _p7 = _mm256_mul_ps(_p7, _mm256_set1_ps(scales[i + ii + 7]));

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);
                __m128i _pp2 = float2int8_avx(_p4, _p6);
                __m128i _pp3 = float2int8_avx(_p5, _p7);

#if __AVX2__
                __m256i _t0 = combine4x2_epi32(_pp0, _pp2);
                __m256i _t1 = combine4x2_epi32(_pp1, _pp3);
                __m256i _t2 = _mm256_unpacklo_epi16(_t0, _t1);
                __m256i _t3 = _mm256_unpackhi_epi16(_t0, _t1);
                _t0 = _mm256_unpacklo_epi32(_t2, _t3);
                _t1 = _mm256_unpackhi_epi32(_t2, _t3);
                _t0 = _mm256_permute4x64_epi64(_t0, _MM_SHUFFLE(3, 1, 2, 0));
                _t1 = _mm256_permute4x64_epi64(_t1, _MM_SHUFFLE(3, 1, 2, 0));

                _mm256_store_si256((__m256i*)pp, _t0);
                _mm256_store_si256((__m256i*)(pp + 32), _t1);
                pp += 64;
#else
                __m128i _tt0 = _mm_unpacklo_epi16(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi16(_pp0, _pp1);
                __m128i _tt2 = _mm_unpacklo_epi16(_pp2, _pp3);
                __m128i _tt3 = _mm_unpackhi_epi16(_pp2, _pp3);
                _pp0 = _mm_unpacklo_epi32(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi32(_tt0, _tt1);
                _pp2 = _mm_unpacklo_epi32(_tt2, _tt3);
                _pp3 = _mm_unpackhi_epi32(_tt2, _tt3);
                __m256i _t0 = combine4x2_epi32(_pp0, _pp1);
                __m256i _t1 = combine4x2_epi32(_pp2, _pp3);
                _mm256_store_si256((__m256i*)pp, _t0);
                _mm256_store_si256((__m256i*)pp1, _t1);
                pp += 32;
                pp1 += 32;
#endif
                p0 += A_hstep * 8;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
        }
        if (elempack == 4)
        {
            __m256 _scales0 = _scales;
            __m256 _scales1 = _scales;
            __m256 _scales2 = _scales;
            __m256 _scales3 = _scales;
            transpose8x4_ps(_scales0, _scales1, _scales2, _scales3);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + 8);
                __m256 _p2 = _mm256_loadu_ps(p0 + 16);
                __m256 _p3 = _mm256_loadu_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scales0);
                _p1 = _mm256_mul_ps(_p1, _scales1);
                _p2 = _mm256_mul_ps(_p2, _scales2);
                _p3 = _mm256_mul_ps(_p3, _scales3);

                __m128i _pp0 = float2int8_avx(_p0, _p1);
                __m128i _pp1 = float2int8_avx(_p2, _p3);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm256_store_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += A_hstep * 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm256_store_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + 8);
                __m256 _p2 = _mm256_loadu_ps(p0 + 16);
                __m256 _p3 = _mm256_loadu_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scales0);
                _p1 = _mm256_mul_ps(_p1, _scales1);
                _p2 = _mm256_mul_ps(_p2, _scales2);
                _p3 = _mm256_mul_ps(_p3, _scales3);

                __m128i _pp0 = float2int8_avx(_p0, _p1);
                __m128i _pp1 = float2int8_avx(_p2, _p3);

#if __AVX2__
                __m128i _t0 = _mm_unpacklo_epi16(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi16(_pp0, _pp1);
                __m128i _t2 = _mm_unpacklo_epi16(_t0, _t1);
                __m128i _t3 = _mm_unpackhi_epi16(_t0, _t1);
                _t0 = _mm_unpacklo_epi16(_t2, _t3);
                _t1 = _mm_unpackhi_epi16(_t2, _t3);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);
                pp += 32;
#else
                __m128i _si = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                __m128i _t0 = _mm_shuffle_epi8(_pp0, _si);
                __m128i _t1 = _mm_shuffle_epi8(_pp1, _si);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)pp1, _t1);
                pp += 16;
                pp1 += 16;
#endif
                p0 += A_hstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + A_hstep);
                __m256 _p2 = _mm256_loadu_ps(p0 + A_hstep * 2);
                __m256 _p3 = _mm256_loadu_ps(p0 + A_hstep * 3);

                _p0 = _mm256_mul_ps(_p0, _scales);
                _p1 = _mm256_mul_ps(_p1, _scales);
                _p2 = _mm256_mul_ps(_p2, _scales);
                _p3 = _mm256_mul_ps(_p3, _scales);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);

                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi16(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi16(_tt0, _tt1);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm256_storeu_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += A_hstep * 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm256_storeu_si256((__m256i*)pp, _w_shift);
                pp += 32;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + A_hstep);

                _p0 = _mm256_mul_ps(_p0, _scales);
                _p1 = _mm256_mul_ps(_p1, _scales);

                __m128i _pp = float2int8_avx(_p0, _p1);

                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);

#if __AVX2__
#if __AVX512F__
                _mm_store_si128((__m128i*)pp, _pp);
#else
                _mm_storeu_si128((__m128i*)pp, _pp);
#endif
                pp += 16;
#else
                _mm_storel_pd((double*)pp, _mm_castsi128_pd(_pp));
                _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_pp));
                pp += 8;
                pp1 += 8;
#endif
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                __m256 _p = _mm256_loadu_ps(p0);

                _p = _mm256_mul_ps(_p, _scales);

                int64_t v = float2int8_avx(_p);

#if __AVX2__
                *(int64_t*)pp = v;
                pp += 8;
#else
                *(int32_t*)pp = (int32_t)v;
                *(int32_t*)pp1 = (int32_t)(v >> 32);
                pp += 4;
                pp1 += 4;
#endif
                p0 += A_hstep;
            }
        }

#if !__AVX2__
        pp = pp1;
        pp1 = pp + max_kk * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m128i _w_shift = _mm_setzero_si128();
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            __m512 _scales0 = _mm512_set1_ps(scales[i + ii]);
            __m512 _scales1 = _mm512_set1_ps(scales[i + ii + 1]);
            __m512 _scales2 = _mm512_set1_ps(scales[i + ii + 2]);
            __m512 _scales3 = _mm512_set1_ps(scales[i + ii + 3]);

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scales0);
                _p1 = _mm512_mul_ps(_p1, _scales1);
                _p2 = _mm512_mul_ps(_p2, _scales2);
                _p3 = _mm512_mul_ps(_p3, _scales3);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp0);
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp1);
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp2);
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp3);

                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);
                _mm_store_si128((__m128i*)(pp + 32), _pp2);
                _mm_store_si128((__m128i*)(pp + 48), _pp3);

                pp += 64;
                p0 += A_hstep * 16;
            }
            if (max_kk >= 4)
            {
                _mm_store_si128((__m128i*)pp, _w_shift);
                pp += 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scales0);
                _p1 = _mm512_mul_ps(_p1, _scales1);
                _p2 = _mm512_mul_ps(_p2, _scales2);
                _p3 = _mm512_mul_ps(_p3, _scales3);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose8x4_epi16(_pp0, _pp1, _pp2, _pp3);

                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);
                _mm_store_si128((__m128i*)(pp + 32), _pp2);
                _mm_store_si128((__m128i*)(pp + 48), _pp3);

                pp += 64;
                p0 += A_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            __m256 _scales0 = _mm256_set1_ps(scales[i + ii]);
            __m256 _scales1 = _mm256_set1_ps(scales[i + ii + 1]);
            __m256 _scales2 = _mm256_set1_ps(scales[i + ii + 2]);
            __m256 _scales3 = _mm256_set1_ps(scales[i + ii + 3]);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scales0);
                _p1 = _mm256_mul_ps(_p1, _scales1);
                _p2 = _mm256_mul_ps(_p2, _scales2);
                _p3 = _mm256_mul_ps(_p3, _scales3);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);

                __m128i _t0 = _mm_unpacklo_epi32(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi32(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi64(_t0, _t1);
                _pp1 = _mm_unpackhi_epi64(_t0, _t1);
#if !__AVXVNNIINT8__
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp0);
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp1);
#endif // !__AVXVNNIINT8__
                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);

                pp += 32;
                p0 += A_hstep * 8;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm_store_si128((__m128i*)pp, _w_shift);
                pp += 16;
            }
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scales0);
                _p1 = _mm256_mul_ps(_p1, _scales1);
                _p2 = _mm256_mul_ps(_p2, _scales2);
                _p3 = _mm256_mul_ps(_p3, _scales3);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);

                __m128i _t0 = _mm_unpacklo_epi16(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi16(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi32(_t0, _t1);
                _pp1 = _mm_unpackhi_epi32(_t0, _t1);

                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);

                pp += 32;
                p0 += A_hstep * 8;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            __m128 _scales0 = _mm_set1_ps(scales[i + ii]);
            __m128 _scales1 = _mm_set1_ps(scales[i + ii + 1]);
            __m128 _scales2 = _mm_set1_ps(scales[i + ii + 2]);
            __m128 _scales3 = _mm_set1_ps(scales[i + ii + 3]);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                __m128 _p2 = _mm_load_ps(p0 + 8);
                __m128 _p3 = _mm_load_ps(p0 + 12);

                _p0 = _mm_mul_ps(_p0, _scales0);
                _p1 = _mm_mul_ps(_p1, _scales1);
                _p2 = _mm_mul_ps(_p2, _scales2);
                _p3 = _mm_mul_ps(_p3, _scales3);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);
#if !__AVXVNNIINT8__
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += A_hstep * 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm_store_si128((__m128i*)pp, _w_shift);
                pp += 16;
            }
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                __m128 _p2 = _mm_load_ps(p0 + 8);
                __m128 _p3 = _mm_load_ps(p0 + 12);

                _p0 = _mm_mul_ps(_p0, _scales0);
                _p1 = _mm_mul_ps(_p1, _scales1);
                _p2 = _mm_mul_ps(_p2, _scales2);
                _p3 = _mm_mul_ps(_p3, _scales3);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);

                _pp = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pp, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                _pp = _mm_shuffle_epi32(_pp, _MM_SHUFFLE(3, 1, 2, 0));

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += A_hstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
        }
        if (elempack == 1)
        {
            __m128 _scales = _mm_load_ps((const float*)scales + i + ii);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + A_hstep * 2);
                __m128 _p3 = _mm_loadu_ps(p0 + A_hstep * 3);

                _p0 = _mm_mul_ps(_p0, _scales);
                _p1 = _mm_mul_ps(_p1, _scales);
                _p2 = _mm_mul_ps(_p2, _scales);
                _p3 = _mm_mul_ps(_p3, _scales);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);

                __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);
#if !__AVXVNNIINT8__
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += A_hstep * 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                _mm_storeu_si128((__m128i*)pp, _w_shift);
                pp += 16;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + A_hstep);
                _p0 = _mm_mul_ps(_p0, _scales);
                _p1 = _mm_mul_ps(_p1, _scales);
                __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
                pp += 8;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p = _mm_loadu_ps(p0);
                _p = _mm_mul_ps(_p, _scales);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            __m512 _scales0 = _mm512_set1_ps(scales[i + ii]);
            __m512 _scales1 = _mm512_set1_ps(scales[i + ii + 1]);

            int kk = 0;
#if __AVX512VNNI__
            __m128i _w_shift = _mm_setzero_si128();
#endif // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);

                _p0 = _mm512_mul_ps(_p0, _scales0);
                _p1 = _mm512_mul_ps(_p1, _scales1);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

#if __AVX512VNNI__
                __m128i _t0 = _mm_unpacklo_epi32(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi32(_pp0, _pp1);

                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _t0);
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _t1);
#else  // __AVX512VNNI__
                __m128i _t0 = _mm_unpacklo_epi16(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi16(_pp0, _pp1);
#endif // __AVX512VNNI__

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += A_hstep * 16;
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                _w_shift = _mm_shuffle_epi32(_w_shift, _MM_SHUFFLE(3, 1, 2, 0));
                _w_shift = _mm_hadd_epi32(_w_shift, _w_shift);
                _mm_storel_epi64((__m128i*)pp, _w_shift);
                pp += 8;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            __m256 _scales0 = _mm256_set1_ps(scales[i + ii]);
            __m256 _scales1 = _mm256_set1_ps(scales[i + ii + 1]);

            int kk = 0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            __m128i _w_shift = _mm_setzero_si128();
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);

                _p0 = _mm256_mul_ps(_p0, _scales0);
                _p1 = _mm256_mul_ps(_p1, _scales1);

                __m128i _pp = float2int8_avx(_p0, _p1);

                _pp = _mm_shuffle_epi32(_pp, _MM_SHUFFLE(3, 1, 2, 0));
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                _pp = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pp, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
#endif // __AVX512VNNI__ || __AVXVNNI__

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += A_hstep * 8;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                _w_shift = _mm_shuffle_epi32(_w_shift, _MM_SHUFFLE(3, 1, 2, 0));
                _w_shift = _mm_hadd_epi32(_w_shift, _w_shift);
                _mm_storel_epi64((__m128i*)pp, _w_shift);
                pp += 8;
            }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        }
#endif // __AVX__
        if (elempack == 4)
        {
            __m128 _scales0 = _mm_set1_ps(scales[i + ii]);
            __m128 _scales1 = _mm_set1_ps(scales[i + ii + 1]);

            int kk = 0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            int w_shift0 = 0;
            int w_shift1 = 0;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                _p0 = _mm_mul_ps(_p0, _scales0);
                _p1 = _mm_mul_ps(_p1, _scales1);
#if __AVX512VNNI__ || __AVXVNNI__
                int64_t v = float2int8_sse(_p0, _p1);
                *(int64_t*)pp = v;
#if !__AVXVNNIINT8__
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128 _t0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                __m128 _t1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 8;
                p0 += A_hstep * 4;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                pp += 8;
            }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const float scale0 = scales[i + ii];
            const float scale1 = scales[i + ii + 1];

            int kk = 0;
#if __SSE2__
            __m128 _scales0 = _mm_set1_ps(scale0);
            __m128 _scales1 = _mm_set1_ps(scale1);
            __m128 _scales0011 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_scales0), _mm_castps_pd(_scales1)));
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            int w_shift0 = 0;
            int w_shift1 = 0;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _p1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                __m128 _p2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2)));
                __m128 _p3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3)));
                __m128 _p01 = _mm_unpacklo_ps(_p0, _p1);
                __m128 _p23 = _mm_unpacklo_ps(_p2, _p3);
                _p0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p01), _mm_castps_pd(_p23)));
                _p1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_p01), _mm_castps_pd(_p23)));
                _p0 = _mm_mul_ps(_p0, _scales0);
                _p1 = _mm_mul_ps(_p1, _scales1);
#if __AVX512VNNI__ || __AVXVNNI__
                int64_t v = float2int8_sse(_p0, _p1);
                *(int64_t*)pp = v;
#if !__AVXVNNIINT8__
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128 _t0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                __m128 _t1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 8;
                p0 += A_hstep * 4;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                pp += 8;
            }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _p1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + A_hstep)));
                __m128 _p = _mm_unpacklo_ps(_p0, _p1);
                _p = _mm_mul_ps(_p, _scales0011);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0 += A_hstep * 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

#if __AVX512VNNI__
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__

        const float scale = scales[i + ii];

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            __m512 _scale = _mm512_set1_ps(scales[i + ii]);

            int kk = 0;
#if __AVX512VNNI__
            __m128i _w_shift = _mm_setzero_si128();
#endif // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p = _mm512_load_ps(p0);
                _p = _mm512_mul_ps(_p, _scale);
                __m128i _pp = float2int8_avx512(_p);
#if __AVX512VNNI__
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _pp);
#endif // __AVX512VNNI__
                _mm_storeu_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += A_hstep * 16;
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = _mm_reduce_add_epi32(_w_shift);
                pp += 4;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            __m256 _scale = _mm256_set1_ps(scales[i + ii]);

            int kk = 0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            int w_shift = 0;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p = _mm256_load_ps(p0);
                _p = _mm256_mul_ps(_p, _scale);
                int64_t v = float2int8_avx(_p);
                *(int64_t*)pp = v;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
                w_shift += pp[4];
                w_shift += pp[5];
                w_shift += pp[6];
                w_shift += pp[7];
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp += 8;
                p0 += A_hstep * 8;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        }
#endif // __AVX__
        if (elempack == 4)
        {
            __m128 _scale = _mm_set1_ps(scales[i + ii]);

            int kk = 0;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            int w_shift = 0;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p = _mm_load_ps(p0);
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp += 4;
                p0 += A_hstep * 4;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
            __m128 _scale = _mm_set1_ps(scales[i + ii]);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            int w_shift = 0;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVX2__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(A_hstep));

                __m128 _p = _mm_i32gather_ps(p0, _vindex, sizeof(float));
#else
                __m128 _p = _mm_setr_ps(p0[0], p0[A_hstep], p0[A_hstep * 2], p0[A_hstep * 3]);
#endif
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp += 4;
                p0 += A_hstep * 4;
            }
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void compute_B_fp32_int8_scale(const Mat& B, float& scale)
{
    // NCNN_LOGE("compute_B_fp32_int8_scale");

    float absmax = 0.f;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _absmax_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _absmax_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _absmax = _mm_setzero_ps();
#endif // __SSE2__
    for (int i = 0; i < (B.dims == 3 ? B.c : B.h); i++)
    {
        const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;
        const float* ptr = (const float*)B + i * B_hstep * B.elempack;

        const int size = B.w * B.elempack;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size; j += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _absmax_avx512 = _mm512_max_ps(_absmax_avx512, abs512_ps(_p));
            ptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size; j += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _absmax_avx = _mm256_max_ps(_absmax_avx, abs256_ps(_p));
            ptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size; j += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _absmax = _mm_max_ps(_absmax, abs_ps(_p));
            ptr += 4;
        }
#endif // __SSE2__
        for (; j < size; j++)
        {
            absmax = std::max(absmax, (float)fabsf(ptr[0]));
            ptr++;
        }
    }
#if __SSE2__
#if __AVX__
#if __AVX512F__
    absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax_avx512));
#endif // __AVX512F__
    absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax_avx));
#endif // __AVX__
    absmax = std::max(absmax, _mm_reduce_max_ps(_absmax));
#endif

    scale = absmax == 0.f ? 1.f : 127.f / absmax;
}

static void pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_B_tile_fp32_to_int8_avx512vnni(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        pack_B_tile_fp32_to_int8_avxvnniint8(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_B_tile_fp32_to_int8_avxvnni(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_B_tile_fp32_to_int8_avx2(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("pack_B_tile_fp32_to_int8 %d %d %d", max_jj, max_kk, elempack);

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

        __m512 _scale = _mm512_set1_ps(scale);
#if __AVX512VNNI__
        __m512i _v127 = _mm512_set1_epi8(127);
#endif // __AVX512VNNI__

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose16x4_epi8(_pp0, _pp1, _pp2, _pp3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _pp = _mm512_add_epi8(_pp, _v127);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

                // transpose16x2_epi8
                __m128i _t0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += 32;
            }
            for (; kk < max_kk; kk++)
            {
                __m512 _p = _mm512_load_ps(p0);

                _p = _mm512_mul_ps(_p, _scale);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + B_hstep * 8);
                __m512 _p3 = _mm512_loadu_ps(p0 + B_hstep * 8 + 16);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                __m128i _t0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi8(_pp0, _pp1);
                __m128i _t2 = _mm_unpacklo_epi8(_pp2, _pp3);
                __m128i _t3 = _mm_unpackhi_epi8(_pp2, _pp3);
                _pp0 = _mm_unpacklo_epi8(_t0, _t1);
                _pp1 = _mm_unpackhi_epi8(_t0, _t1);
                _pp2 = _mm_unpacklo_epi8(_t2, _t3);
                _pp3 = _mm_unpackhi_epi8(_t2, _t3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _pp = _mm512_add_epi8(_pp, _v127);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + B_hstep * 8);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp0 = _mm_shuffle_epi8(_pp0, _si);
                _pp1 = _mm_shuffle_epi8(_pp1, _si);

                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);

                pp += 32;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + B_hstep * 8);

                __m512 _p = combine8x2_ps(_p0, _p1);
                _p = _mm512_mul_ps(_p, _scale);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + B_hstep * 4);
                __m512 _p2 = _mm512_loadu_ps(p0 + B_hstep * 8);
                __m512 _p3 = _mm512_loadu_ps(p0 + B_hstep * 12);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _pp = _mm512_add_epi8(_pp, _v127);

                __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                _pp = _mm512_shuffle_epi8(_pp, _mm512_broadcast_i32x4(_si));

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + B_hstep * 4);
                __m256 _p2 = _mm256_loadu_ps(p0 + B_hstep * 8);
                __m256 _p3 = _mm256_loadu_ps(p0 + B_hstep * 12);

                __m512 _p01 = combine8x2_ps(_p0, _p1);
                __m512 _p23 = combine8x2_ps(_p2, _p3);

                _p01 = _mm512_mul_ps(_p01, _scale);
                _p23 = _mm512_mul_ps(_p23, _scale);

                __m128i _pp0 = float2int8_avx512(_p01);
                __m128i _pp1 = float2int8_avx512(_p23);

                __m128i _si = _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
                _pp0 = _mm_shuffle_epi8(_pp0, _si);
                _pp1 = _mm_shuffle_epi8(_pp1, _si);

                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);

                pp += 32;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + B_hstep * 4);
                __m128 _p2 = _mm_loadu_ps(p0 + B_hstep * 8);
                __m128 _p3 = _mm_loadu_ps(p0 + B_hstep * 12);

                __m512 _p = combine4x4_ps(_p0, _p1, _p2, _p3);
                _p = _mm512_mul_ps(_p, _scale);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + B_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + B_hstep * 2);
                __m128 _p3 = _mm_loadu_ps(p0 + B_hstep * 3);
                __m128 _p4 = _mm_loadu_ps(p0 + B_hstep * 4);
                __m128 _p5 = _mm_loadu_ps(p0 + B_hstep * 5);
                __m128 _p6 = _mm_loadu_ps(p0 + B_hstep * 6);
                __m128 _p7 = _mm_loadu_ps(p0 + B_hstep * 7);
                __m128 _p8 = _mm_loadu_ps(p0 + B_hstep * 8);
                __m128 _p9 = _mm_loadu_ps(p0 + B_hstep * 9);
                __m128 _pa = _mm_loadu_ps(p0 + B_hstep * 10);
                __m128 _pb = _mm_loadu_ps(p0 + B_hstep * 11);
                __m128 _pc = _mm_loadu_ps(p0 + B_hstep * 12);
                __m128 _pd = _mm_loadu_ps(p0 + B_hstep * 13);
                __m128 _pe = _mm_loadu_ps(p0 + B_hstep * 14);
                __m128 _pf = _mm_loadu_ps(p0 + B_hstep * 15);

                __m512 _t0 = combine4x4_ps(_p0, _p1, _p2, _p3);
                __m512 _t1 = combine4x4_ps(_p4, _p5, _p6, _p7);
                __m512 _t2 = combine4x4_ps(_p8, _p9, _pa, _pb);
                __m512 _t3 = combine4x4_ps(_pc, _pd, _pe, _pf);

                _t0 = _mm512_mul_ps(_t0, _scale);
                _t1 = _mm512_mul_ps(_t1, _scale);
                _t2 = _mm512_mul_ps(_t2, _scale);
                _t3 = _mm512_mul_ps(_t3, _scale);

                __m128i _pp0 = float2int8_avx512(_t0);
                __m128i _pp1 = float2int8_avx512(_t1);
                __m128i _pp2 = float2int8_avx512(_t2);
                __m128i _pp3 = float2int8_avx512(_t3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _pp = _mm512_add_epi8(_pp, _v127);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(B_hstep));

                __m512 _p0 = _mm512_i32gather_ps(_vindex, p0, sizeof(float));
                __m512 _p1 = _mm512_i32gather_ps(_vindex, p0 + 1, sizeof(float));
                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

                __m128i _t0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(B_hstep));

                __m512 _p = _mm512_i32gather_ps(_vindex, p0, sizeof(float));
                _p = _mm512_mul_ps(_p, _scale);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0++;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

#if __AVX__
        __m256 _scale = _mm256_set1_ps(scale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m256i _v127 = _mm256_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
#else
        __m128 _scale = _mm_set1_ps(scale);
#endif // __AVX__

#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);

                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi16(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi16(_tt0, _tt1);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _pp = _mm256_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                _mm256_storeu_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += 32;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);

                __m128i _pp = float2int8_avx(_p0, _p1);

                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);

#if __AVX512F__
                _mm_store_si128((__m128i*)pp, _pp);
#else
                _mm_storeu_si128((__m128i*)pp, _pp);
#endif
                pp += 16;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                __m256 _p = _mm256_load_ps(p0);

                _p = _mm256_mul_ps(_p, _scale);

                int64_t v = float2int8_avx(_p);

                *(int64_t*)pp = v;
                pp += 8;
                p0 += 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + 8);
                __m256 _p2 = _mm256_loadu_ps(p0 + B_hstep * 4);
                __m256 _p3 = _mm256_loadu_ps(p0 + B_hstep * 4 + 8);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p1);
                __m128i _pp1 = float2int8_avx(_p2, _p3);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _pp = _mm256_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                _pp = _mm256_shuffle_epi8(_pp, combine4x2_epi32(_si, _si));

                _mm256_storeu_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += 16;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX__
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + B_hstep * 4);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);

                __m128i _pp = float2int8_avx(_p0, _p1);

                __m128i _si = _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);
#else  // __AVX__
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                __m128 _p2 = _mm_load_ps(p0 + B_hstep * 4);
                __m128 _p3 = _mm_load_ps(p0 + B_hstep * 4 + 4);

                __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                __m128 _t2 = _mm_unpacklo_ps(_p2, _p3);
                __m128 _t3 = _mm_unpackhi_ps(_p2, _p3);

                _t0 = _mm_mul_ps(_t0, _scale);
                _t1 = _mm_mul_ps(_t1, _scale);
                _t2 = _mm_mul_ps(_t2, _scale);
                _t3 = _mm_mul_ps(_t3, _scale);

                __m128i _pp = float2int8_sse(_t0, _t1, _t2, _t3);
#endif // __AVX__

#if __AVX512F__
                _mm_store_si128((__m128i*)pp, _pp);
#else
                _mm_storeu_si128((__m128i*)pp, _pp);
#endif
                pp += 16;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + B_hstep * 4);

#if __AVX__
                __m256 _p = combine4x2_ps(_p0, _p1);
                _p = _mm256_mul_ps(_p, _scale);

                int64_t v = float2int8_avx(_p);
#else  // __AVX__
                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);

                int64_t v = float2int8_sse(_p0, _p1);
#endif // __AVX__

                *(int64_t*)pp = v;
                pp += 8;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + B_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + B_hstep * 2);
                __m128 _p3 = _mm_loadu_ps(p0 + B_hstep * 3);
                __m128 _p4 = _mm_loadu_ps(p0 + B_hstep * 4);
                __m128 _p5 = _mm_loadu_ps(p0 + B_hstep * 5);
                __m128 _p6 = _mm_loadu_ps(p0 + B_hstep * 6);
                __m128 _p7 = _mm_loadu_ps(p0 + B_hstep * 7);

                __m256 _t0 = combine4x2_ps(_p0, _p1);
                __m256 _t1 = combine4x2_ps(_p2, _p3);
                __m256 _t2 = combine4x2_ps(_p4, _p5);
                __m256 _t3 = combine4x2_ps(_p6, _p7);

                _t0 = _mm256_mul_ps(_t0, _scale);
                _t1 = _mm256_mul_ps(_t1, _scale);
                _t2 = _mm256_mul_ps(_t2, _scale);
                _t3 = _mm256_mul_ps(_t3, _scale);

                __m128i _pp0 = float2int8_avx(_t0, _t1);
                __m128i _pp1 = float2int8_avx(_t2, _t3);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _pp = _mm256_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                _mm256_storeu_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX__
#if __AVX2__
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(B_hstep));

                __m256 _p0 = _mm256_i32gather_ps(p0, _vindex, sizeof(float));
                __m256 _p1 = _mm256_i32gather_ps(p0 + 1, _vindex, sizeof(float));
#else
                __m256 _p0 = _mm256_setr_ps(p0[0], p0[1], p0[B_hstep], p0[B_hstep + 1], p0[B_hstep * 2], p0[B_hstep * 2 + 1], p0[B_hstep * 3], p0[B_hstep * 3 + 1]);
                __m256 _p1 = _mm256_setr_ps(p0[B_hstep * 4], p0[B_hstep * 4 + 1], p0[B_hstep * 5], p0[B_hstep * 5 + 1], p0[B_hstep * 6], p0[B_hstep * 6 + 1], p0[B_hstep * 7], p0[B_hstep * 7 + 1]);
#endif

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);

                __m128i _pp = float2int8_avx(_p0, _p1);

#if __AVX2__
                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);
#endif
#else  // __AVX__
                __m128 _p0 = _mm_setr_ps(p0[0], p0[1], p0[B_hstep], p0[B_hstep + 1]);
                __m128 _p1 = _mm_setr_ps(p0[B_hstep * 2], p0[B_hstep * 2 + 1], p0[B_hstep * 3], p0[B_hstep * 3 + 1]);
                __m128 _p2 = _mm_setr_ps(p0[B_hstep * 4], p0[B_hstep * 4 + 1], p0[B_hstep * 5], p0[B_hstep * 5 + 1]);
                __m128 _p3 = _mm_setr_ps(p0[B_hstep * 6], p0[B_hstep * 6 + 1], p0[B_hstep * 7], p0[B_hstep * 7 + 1]);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                _p2 = _mm_mul_ps(_p2, _scale);
                _p3 = _mm_mul_ps(_p3, _scale);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);
#endif // __AVX__

#if __AVX512F__
                _mm_store_si128((__m128i*)pp, _pp);
#else
                _mm_storeu_si128((__m128i*)pp, _pp);
#endif
                pp += 16;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
#if __AVX__
#if __AVX2__
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(B_hstep));

                __m256 _p = _mm256_i32gather_ps(p0, _vindex, sizeof(float));
#else
                __m256 _p = _mm256_setr_ps(p0[0], p0[B_hstep], p0[B_hstep * 2], p0[B_hstep * 3], p0[B_hstep * 4], p0[B_hstep * 5], p0[B_hstep * 6], p0[B_hstep * 7]);
#endif

                _p = _mm256_mul_ps(_p, _scale);

                int64_t v = float2int8_avx(_p);
#else  // __AVX__
                __m128 _p0 = _mm_setr_ps(p0[0], p0[B_hstep], p0[B_hstep * 2], p0[B_hstep * 3]);
                __m128 _p1 = _mm_setr_ps(p0[B_hstep * 4], p0[B_hstep * 5], p0[B_hstep * 6], p0[B_hstep * 7]);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);

                int64_t v = float2int8_sse(_p0, _p1);
#endif // __AVX__

                *(int64_t*)pp = v;
                pp += 8;
                p0++;
            }
        }
    }
#else // defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512 _scale = _mm512_set1_ps(scale);
#if __AVX512VNNI__ || __AVXVNNI__
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__ || __AVXVNNI__

        for (; jj + 15 < max_jj; jj += 16)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

            signed char* pp1 = pp + max_kk * 4;
            signed char* pp2 = pp + max_kk * 8;
            signed char* pp3 = pp + max_kk * 12;

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                _pp0 = _mm_add_epi8(_pp0, _v127);
                _pp1 = _mm_add_epi8(_pp1, _v127);
                _pp2 = _mm_add_epi8(_pp2, _v127);
                _pp3 = _mm_add_epi8(_pp3, _v127);

                transpose16x4_epi8(_pp0, _pp1, _pp2, _pp3);

                _mm_storeu_si128((__m128i*)pp, _pp0);
                _mm_storeu_si128((__m128i*)pp1, _pp1);
                _mm_storeu_si128((__m128i*)pp2, _pp2);
                _mm_storeu_si128((__m128i*)pp3, _pp3);

                pp += 16;
                pp1 += 16;
                pp2 += 16;
                pp3 += 16;
                p0 += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

                __m128i _t0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_storel_pd((double*)pp, _mm_castsi128_pd(_t0));
                _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_t0));
                _mm_storel_pd((double*)pp2, _mm_castsi128_pd(_t1));
                _mm_storeh_pd((double*)pp3, _mm_castsi128_pd(_t1));

                pp += 8;
                pp1 += 8;
                pp2 += 8;
                pp3 += 8;
                p0 += 32;
            }
            for (; kk < max_kk; kk++)
            {
                __m512 _p = _mm512_load_ps(p0);

                _p = _mm512_mul_ps(_p, _scale);

                __m128i _v = float2int8_avx512(_p);

                *(int*)pp = _mm_extract_epi32(_v, 0);
                *(int*)pp1 = _mm_extract_epi32(_v, 1);
                *(int*)pp2 = _mm_extract_epi32(_v, 2);
                *(int*)pp3 = _mm_extract_epi32(_v, 3);

                pp += 4;
                pp1 += 4;
                pp2 += 4;
                pp3 += 4;
                p0 += 16;
            }

            pp = pp3;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        __m256 _scale = _mm256_set1_ps(scale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

            signed char* pp1 = pp + max_kk * 4;

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);
#if !__AVXVNNIINT8__
                _pp0 = _mm_add_epi8(_pp0, _v127);
                _pp1 = _mm_add_epi8(_pp1, _v127);
#endif // !__AVXVNNIINT8__
                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi16(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi16(_tt0, _tt1);

                _mm_storeu_si128((__m128i*)pp, _pp0);
                _mm_storeu_si128((__m128i*)pp1, _pp1);

                pp += 16;
                pp1 += 16;
                p0 += 32;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);

                __m128i _pp = float2int8_avx(_p0, _p1);

                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);

                _mm_storel_pd((double*)pp, _mm_castsi128_pd(_pp));
                _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_pp));
                pp += 8;
                pp1 += 8;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                __m256 _p = _mm256_load_ps(p0);

                _p = _mm256_mul_ps(_p, _scale);

                int64_t v = float2int8_avx(_p);

                *(int32_t*)pp = (int32_t)v;
                *(int32_t*)pp1 = (int32_t)(v >> 32);

                pp += 4;
                pp1 += 4;
                p0 += 8;
            }

            pp = pp1;
        }
    }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

        __m128 _scale = _mm_set1_ps(scale);
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                __m128 _p2 = _mm_load_ps(p0 + 8);
                __m128 _p3 = _mm_load_ps(p0 + 12);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                _p2 = _mm_mul_ps(_p2, _scale);
                _p3 = _mm_mul_ps(_p3, _scale);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);

                __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);
#if !__AVXVNNIINT8__
                _pp = _mm_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 16;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p = _mm_load_ps(p0);
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + B_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + B_hstep * 2);
                __m128 _p3 = _mm_loadu_ps(p0 + B_hstep * 3);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                _p2 = _mm_mul_ps(_p2, _scale);
                _p3 = _mm_mul_ps(_p3, _scale);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);
#if !__AVXVNNIINT8__
                _pp = _mm_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX2__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(B_hstep));

                __m128 _t0 = _mm_i32gather_ps(p0, _vindex, sizeof(float));
                __m128 _t1 = _mm_i32gather_ps(p0 + 1, _vindex, sizeof(float));
                __m128 _p0 = _mm_unpacklo_ps(_t0, _t1);
                __m128 _p1 = _mm_unpackhi_ps(_t0, _t1);
#else
                __m128 _p0 = _mm_setr_ps(p0[0], p0[1], p0[B_hstep], p0[B_hstep + 1]);
                __m128 _p1 = _mm_setr_ps(p0[B_hstep * 2], p0[B_hstep * 2 + 1], p0[B_hstep * 3], p0[B_hstep * 3 + 1]);
#endif
                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                int64_t v = float2int8_sse(_p0, _p1);
                *(int64_t*)pp = v;
                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
#if __AVX2__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(B_hstep));

                __m128 _p = _mm_i32gather_ps(p0, _vindex, sizeof(float));
#else
                __m128 _p = _mm_setr_ps(p0[0], p0[B_hstep], p0[B_hstep * 2], p0[B_hstep * 3]);
#endif
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0++;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

#if __SSE2__
        __m128 _scale = _mm_set1_ps(scale);
#endif // __SSE2__

        // if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + B_hstep);
                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
#if __AVX512VNNI__ || __AVXVNNI__
                int64_t v = float2int8_sse(_p0, _p1);
                *(int64_t*)pp = v;
#if !__AVXVNNIINT8__
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
                pp[4] += 127;
                pp[5] += 127;
                pp[6] += 127;
                pp[7] += 127;
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128 _t0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                __m128 _t1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 8;
                p0 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _p1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + B_hstep)));
                __m128 _p = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0 += 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp += 2;
                p0++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

#if __SSE2__
        __m128 _scale = _mm_set1_ps(scale);
#endif // __SSE2__

        // if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p = _mm_loadu_ps(p0);
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp += 4;
                p0 += 4;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_pack_B_tile_fp32_to_int8_avx512vnni(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        transpose_pack_B_tile_fp32_to_int8_avxvnniint8(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_pack_B_tile_fp32_to_int8_avxvnni(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_pack_B_tile_fp32_to_int8_avx2(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("transpose_pack_B_tile_fp32_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

        __m512 _scale = _mm512_set1_ps(scale);
#if __AVX512VNNI__
        __m512i _v127 = _mm512_set1_epi8(127);
#endif // __AVX512VNNI__

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);
                __m512 _p4 = _mm512_load_ps(p0 + 64);
                __m512 _p5 = _mm512_load_ps(p0 + 80);
                __m512 _p6 = _mm512_load_ps(p0 + 96);
                __m512 _p7 = _mm512_load_ps(p0 + 112);
                __m512 _p8 = _mm512_load_ps(p0 + 128);
                __m512 _p9 = _mm512_load_ps(p0 + 128 + 16);
                __m512 _pa = _mm512_load_ps(p0 + 128 + 32);
                __m512 _pb = _mm512_load_ps(p0 + 128 + 48);
                __m512 _pc = _mm512_load_ps(p0 + 128 + 64);
                __m512 _pd = _mm512_load_ps(p0 + 128 + 80);
                __m512 _pe = _mm512_load_ps(p0 + 128 + 96);
                __m512 _pf = _mm512_load_ps(p0 + 128 + 112);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);
                _p4 = _mm512_mul_ps(_p4, _scale);
                _p5 = _mm512_mul_ps(_p5, _scale);
                _p6 = _mm512_mul_ps(_p6, _scale);
                _p7 = _mm512_mul_ps(_p7, _scale);
                _p8 = _mm512_mul_ps(_p8, _scale);
                _p9 = _mm512_mul_ps(_p9, _scale);
                _pa = _mm512_mul_ps(_pa, _scale);
                _pb = _mm512_mul_ps(_pb, _scale);
                _pc = _mm512_mul_ps(_pc, _scale);
                _pd = _mm512_mul_ps(_pd, _scale);
                _pe = _mm512_mul_ps(_pe, _scale);
                _pf = _mm512_mul_ps(_pf, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);
                __m128i _pp8 = float2int8_avx512(_p8);
                __m128i _pp9 = float2int8_avx512(_p9);
                __m128i _ppa = float2int8_avx512(_pa);
                __m128i _ppb = float2int8_avx512(_pb);
                __m128i _ppc = float2int8_avx512(_pc);
                __m128i _ppd = float2int8_avx512(_pd);
                __m128i _ppe = float2int8_avx512(_pe);
                __m128i _ppf = float2int8_avx512(_pf);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp4, _pp8, _ppc);
                __m512i _t1 = combine4x4_epi32(_pp1, _pp5, _pp9, _ppd);
                __m512i _t2 = combine4x4_epi32(_pp2, _pp6, _ppa, _ppe);
                __m512i _t3 = combine4x4_epi32(_pp3, _pp7, _ppb, _ppf);

                __m512i _t4 = _mm512_unpacklo_epi32(_t0, _t1);
                __m512i _t5 = _mm512_unpackhi_epi32(_t0, _t1);
                __m512i _t6 = _mm512_unpacklo_epi32(_t2, _t3);
                __m512i _t7 = _mm512_unpackhi_epi32(_t2, _t3);
                _t0 = _mm512_unpacklo_epi64(_t4, _t6);
                _t1 = _mm512_unpackhi_epi64(_t4, _t6);
                _t2 = _mm512_unpacklo_epi64(_t5, _t7);
                _t3 = _mm512_unpackhi_epi64(_t5, _t7);

                _t0 = _mm512_add_epi8(_t0, _v127);
                _t1 = _mm512_add_epi8(_t1, _v127);
                _t2 = _mm512_add_epi8(_t2, _v127);
                _t3 = _mm512_add_epi8(_t3, _v127);

                _mm512_store_si512((__m512i*)pp, _t0);
                _mm512_store_si512((__m512i*)(pp + 64), _t1);
                _mm512_store_si512((__m512i*)(pp + 128), _t2);
                _mm512_store_si512((__m512i*)(pp + 192), _t3);

                pp += 256;
                p0 += B_hstep * 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);
                __m512 _p4 = _mm512_load_ps(p0 + 64);
                __m512 _p5 = _mm512_load_ps(p0 + 80);
                __m512 _p6 = _mm512_load_ps(p0 + 96);
                __m512 _p7 = _mm512_load_ps(p0 + 112);
                __m512 _p8 = _mm512_load_ps(p0 + 128);
                __m512 _p9 = _mm512_load_ps(p0 + 128 + 16);
                __m512 _pa = _mm512_load_ps(p0 + 128 + 32);
                __m512 _pb = _mm512_load_ps(p0 + 128 + 48);
                __m512 _pc = _mm512_load_ps(p0 + 128 + 64);
                __m512 _pd = _mm512_load_ps(p0 + 128 + 80);
                __m512 _pe = _mm512_load_ps(p0 + 128 + 96);
                __m512 _pf = _mm512_load_ps(p0 + 128 + 112);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);
                _p4 = _mm512_mul_ps(_p4, _scale);
                _p5 = _mm512_mul_ps(_p5, _scale);
                _p6 = _mm512_mul_ps(_p6, _scale);
                _p7 = _mm512_mul_ps(_p7, _scale);
                _p8 = _mm512_mul_ps(_p8, _scale);
                _p9 = _mm512_mul_ps(_p9, _scale);
                _pa = _mm512_mul_ps(_pa, _scale);
                _pb = _mm512_mul_ps(_pb, _scale);
                _pc = _mm512_mul_ps(_pc, _scale);
                _pd = _mm512_mul_ps(_pd, _scale);
                _pe = _mm512_mul_ps(_pe, _scale);
                _pf = _mm512_mul_ps(_pf, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);
                __m128i _pp8 = float2int8_avx512(_p8);
                __m128i _pp9 = float2int8_avx512(_p9);
                __m128i _ppa = float2int8_avx512(_pa);
                __m128i _ppb = float2int8_avx512(_pb);
                __m128i _ppc = float2int8_avx512(_pc);
                __m128i _ppd = float2int8_avx512(_pd);
                __m128i _ppe = float2int8_avx512(_pe);
                __m128i _ppf = float2int8_avx512(_pf);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp4, _pp8, _ppc);
                __m512i _t1 = combine4x4_epi32(_pp1, _pp5, _pp9, _ppd);
                __m512i _t2 = combine4x4_epi32(_pp2, _pp6, _ppa, _ppe);
                __m512i _t3 = combine4x4_epi32(_pp3, _pp7, _ppb, _ppf);

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

                _mm512_store_si512((__m512i*)pp, _t0);
                _mm512_store_si512((__m512i*)(pp + 64), _t1);
                _mm512_store_si512((__m512i*)(pp + 128), _t2);
                _mm512_store_si512((__m512i*)(pp + 192), _t3);

                pp += 256;
                p0 += B_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + 32);
                __m512 _p3 = _mm512_loadu_ps(p0 + 48);
                __m512 _p4 = _mm512_loadu_ps(p0 + 64);
                __m512 _p5 = _mm512_loadu_ps(p0 + 80);
                __m512 _p6 = _mm512_loadu_ps(p0 + 96);
                __m512 _p7 = _mm512_loadu_ps(p0 + 112);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);
                _p4 = _mm512_mul_ps(_p4, _scale);
                _p5 = _mm512_mul_ps(_p5, _scale);
                _p6 = _mm512_mul_ps(_p6, _scale);
                _p7 = _mm512_mul_ps(_p7, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp2, _pp4, _pp6);
                __m512i _t1 = combine4x4_epi32(_pp1, _pp3, _pp5, _pp7);

                __m512i _t2 = _mm512_unpacklo_epi32(_t0, _t1);
                __m512i _t3 = _mm512_unpackhi_epi32(_t0, _t1);
                __m512i _ppa = _mm512_unpacklo_epi32(_t2, _t3);
                __m512i _ppb = _mm512_unpackhi_epi32(_t2, _t3);

                _ppa = _mm512_add_epi8(_ppa, _v127);
                _ppb = _mm512_add_epi8(_ppb, _v127);

                _mm512_store_si512((__m512i*)pp, _ppa);
                _mm512_store_si512((__m512i*)(pp + 64), _ppb);

                pp += 128;
                p0 += B_hstep * 8;
            }
#else  // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + 32);
                __m512 _p3 = _mm512_loadu_ps(p0 + 48);
                __m512 _p4 = _mm512_loadu_ps(p0 + 64);
                __m512 _p5 = _mm512_loadu_ps(p0 + 80);
                __m512 _p6 = _mm512_loadu_ps(p0 + 96);
                __m512 _p7 = _mm512_loadu_ps(p0 + 112);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);
                _p4 = _mm512_mul_ps(_p4, _scale);
                _p5 = _mm512_mul_ps(_p5, _scale);
                _p6 = _mm512_mul_ps(_p6, _scale);
                _p7 = _mm512_mul_ps(_p7, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp2, _pp4, _pp6);
                __m512i _t1 = combine4x4_epi32(_pp1, _pp3, _pp5, _pp7);

                __m512i _t2 = _mm512_unpacklo_epi16(_t0, _t1);
                __m512i _t3 = _mm512_unpackhi_epi16(_t0, _t1);
                _t0 = _mm512_unpacklo_epi16(_t2, _t3);
                _t1 = _mm512_unpackhi_epi16(_t2, _t3);
                _t0 = _mm512_permutex_epi64(_t0, _MM_SHUFFLE(3, 1, 2, 0));
                _t1 = _mm512_permutex_epi64(_t1, _MM_SHUFFLE(3, 1, 2, 0));
                __m512i _ppa = _mm512_shuffle_i32x4(_t0, _t0, _MM_SHUFFLE(3, 1, 2, 0));
                __m512i _ppb = _mm512_shuffle_i32x4(_t1, _t1, _MM_SHUFFLE(3, 1, 2, 0));

                _mm512_store_si512((__m512i*)pp, _ppa);
                _mm512_store_si512((__m512i*)(pp + 64), _ppb);

                pp += 128;
                p0 += B_hstep * 8;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + 32);
                __m512 _p3 = _mm512_loadu_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _pp = _mm512_add_epi8(_pp, _v127);

                _mm512_store_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += B_hstep * 4;
            }
#else  // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + 16);
                __m512 _p2 = _mm512_loadu_ps(p0 + 32);
                __m512 _p3 = _mm512_loadu_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                __m256i _pp02 = combine4x2_epi32(_pp0, _pp2);
                __m256i _pp13 = combine4x2_epi32(_pp1, _pp3);

                __m256i _t0 = _mm256_unpacklo_epi16(_pp02, _pp13);
                __m256i _t1 = _mm256_unpackhi_epi16(_pp02, _pp13);
                __m256i _t2 = _mm256_unpacklo_epi16(_t0, _t1);
                __m256i _t3 = _mm256_unpackhi_epi16(_t0, _t1);
                _t0 = _mm256_unpacklo_epi16(_t2, _t3);
                _t1 = _mm256_unpackhi_epi16(_t2, _t3);

                _mm256_store_si256((__m256i*)pp, _t0);
                _mm256_store_si256((__m256i*)(pp + 32), _t1);

                pp += 64;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + B_hstep);
                __m512 _p2 = _mm512_loadu_ps(p0 + B_hstep * 2);
                __m512 _p3 = _mm512_loadu_ps(p0 + B_hstep * 3);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose16x4_epi8(_pp0, _pp1, _pp2, _pp3);

                __m512i _pp = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _pp = _mm512_add_epi8(_pp, _v127);

                _mm512_storeu_si512((__m512i*)pp, _pp);

                pp += 64;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512 _p0 = _mm512_loadu_ps(p0);
                __m512 _p1 = _mm512_loadu_ps(p0 + B_hstep);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

                // transpose16x2_epi8
                __m128i _t0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi8(_pp0, _pp1);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                __m512 _p = _mm512_loadu_ps(p0);

                _p = _mm512_mul_ps(_p, _scale);

                __m128i _pp = float2int8_avx512(_p);

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += B_hstep;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m256i _v127 = _mm256_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            __m512 _scale = _mm512_set1_ps(scale);

            int kk = 0;
#if __AVX512VNNI__
            __m512i _v127_avx512 = _mm512_set1_epi8(127);
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);
                __m512 _p4 = _mm512_load_ps(p0 + 64);
                __m512 _p5 = _mm512_load_ps(p0 + 80);
                __m512 _p6 = _mm512_load_ps(p0 + 96);
                __m512 _p7 = _mm512_load_ps(p0 + 112);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);
                _p4 = _mm512_mul_ps(_p4, _scale);
                _p5 = _mm512_mul_ps(_p5, _scale);
                _p6 = _mm512_mul_ps(_p6, _scale);
                _p7 = _mm512_mul_ps(_p7, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);

                transpose4x8_epi32(_pp0, _pp1, _pp2, _pp3, _pp4, _pp5, _pp6, _pp7);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);
                __m512i _t1 = combine4x4_epi32(_pp4, _pp5, _pp6, _pp7);

                _t0 = _mm512_add_epi8(_t0, _v127_avx512);
                _t1 = _mm512_add_epi8(_t1, _v127_avx512);

                _mm512_store_si512((__m512i*)pp, _t0);
                _mm512_store_si512((__m512i*)(pp + 64), _t1);

                pp += 128;
                p0 += B_hstep * 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);
                __m512 _p4 = _mm512_load_ps(p0 + 64);
                __m512 _p5 = _mm512_load_ps(p0 + 80);
                __m512 _p6 = _mm512_load_ps(p0 + 96);
                __m512 _p7 = _mm512_load_ps(p0 + 112);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);
                _p4 = _mm512_mul_ps(_p4, _scale);
                _p5 = _mm512_mul_ps(_p5, _scale);
                _p6 = _mm512_mul_ps(_p6, _scale);
                _p7 = _mm512_mul_ps(_p7, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);
                __m128i _pp4 = float2int8_avx512(_p4);
                __m128i _pp5 = float2int8_avx512(_p5);
                __m128i _pp6 = float2int8_avx512(_p6);
                __m128i _pp7 = float2int8_avx512(_p7);

                transpose8x8_epi16(_pp0, _pp1, _pp2, _pp3, _pp4, _pp5, _pp6, _pp7);

                __m512i _t0 = combine4x4_epi32(_pp0, _pp1, _pp2, _pp3);
                __m512i _t1 = combine4x4_epi32(_pp4, _pp5, _pp6, _pp7);

                _mm512_store_si512((__m512i*)pp, _t0);
                _mm512_store_si512((__m512i*)(pp + 64), _t1);

                pp += 128;
                p0 += B_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            __m256 _scale = _mm256_set1_ps(scale);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);
                __m256 _p4 = _mm256_load_ps(p0 + 32);
                __m256 _p5 = _mm256_load_ps(p0 + 40);
                __m256 _p6 = _mm256_load_ps(p0 + 48);
                __m256 _p7 = _mm256_load_ps(p0 + 56);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);
                _p4 = _mm256_mul_ps(_p4, _scale);
                _p5 = _mm256_mul_ps(_p5, _scale);
                _p6 = _mm256_mul_ps(_p6, _scale);
                _p7 = _mm256_mul_ps(_p7, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);
                __m128i _pp2 = float2int8_avx(_p4, _p6);
                __m128i _pp3 = float2int8_avx(_p5, _p7);

                __m256i _t0 = combine4x2_epi32(_pp0, _pp2);
                __m256i _t1 = combine4x2_epi32(_pp1, _pp3);

                __m256i _t2 = _mm256_unpacklo_epi32(_t0, _t1);
                __m256i _t3 = _mm256_unpackhi_epi32(_t0, _t1);
                _t0 = _mm256_unpacklo_epi64(_t2, _t3);
                _t1 = _mm256_unpackhi_epi64(_t2, _t3);
#if !__AVXVNNIINT8__
                _t0 = _mm256_add_epi8(_t0, _v127);
                _t1 = _mm256_add_epi8(_t1, _v127);
#endif // !__AVXVNNIINT8__
                _mm256_store_si256((__m256i*)pp, _t0);
                _mm256_store_si256((__m256i*)(pp + 32), _t1);

                pp += 64;
                p0 += B_hstep * 8;
            }
#else // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);
                __m256 _p4 = _mm256_load_ps(p0 + 32);
                __m256 _p5 = _mm256_load_ps(p0 + 40);
                __m256 _p6 = _mm256_load_ps(p0 + 48);
                __m256 _p7 = _mm256_load_ps(p0 + 56);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);
                _p4 = _mm256_mul_ps(_p4, _scale);
                _p5 = _mm256_mul_ps(_p5, _scale);
                _p6 = _mm256_mul_ps(_p6, _scale);
                _p7 = _mm256_mul_ps(_p7, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);
                __m128i _pp2 = float2int8_avx(_p4, _p6);
                __m128i _pp3 = float2int8_avx(_p5, _p7);

#if __AVX2__
                __m256i _t0 = combine4x2_epi32(_pp0, _pp2);
                __m256i _t1 = combine4x2_epi32(_pp1, _pp3);
                __m256i _t2 = _mm256_unpacklo_epi16(_t0, _t1);
                __m256i _t3 = _mm256_unpackhi_epi16(_t0, _t1);
                _t0 = _mm256_unpacklo_epi32(_t2, _t3);
                _t1 = _mm256_unpackhi_epi32(_t2, _t3);
                _t0 = _mm256_permute4x64_epi64(_t0, _MM_SHUFFLE(3, 1, 2, 0));
                _t1 = _mm256_permute4x64_epi64(_t1, _MM_SHUFFLE(3, 1, 2, 0));
#else
                __m128i _tt0 = _mm_unpacklo_epi16(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi16(_pp0, _pp1);
                __m128i _tt2 = _mm_unpacklo_epi16(_pp2, _pp3);
                __m128i _tt3 = _mm_unpackhi_epi16(_pp2, _pp3);
                _pp0 = _mm_unpacklo_epi32(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi32(_tt0, _tt1);
                _pp2 = _mm_unpacklo_epi32(_tt2, _tt3);
                _pp3 = _mm_unpackhi_epi32(_tt2, _tt3);
                _tt0 = _mm_unpacklo_epi64(_pp0, _pp2);
                _tt1 = _mm_unpackhi_epi64(_pp0, _pp2);
                _tt2 = _mm_unpacklo_epi64(_pp1, _pp3);
                _tt3 = _mm_unpackhi_epi64(_pp1, _pp3);
                __m256i _t0 = combine4x2_epi32(_tt0, _tt1);
                __m256i _t1 = combine4x2_epi32(_tt2, _tt3);
#endif
                _mm256_store_si256((__m256i*)pp, _t0);
                _mm256_store_si256((__m256i*)(pp + 32), _t1);

                pp += 64;
                p0 += B_hstep * 8;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
        }
#endif // __AVX__
        if (elempack == 4)
        {
#if __AVX__
            __m256 _scale = _mm256_set1_ps(scale);
#else
            __m128 _scale = _mm_set1_ps(scale);
#endif

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + 8);
                __m256 _p2 = _mm256_loadu_ps(p0 + 16);
                __m256 _p3 = _mm256_loadu_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p1);
                __m128i _pp1 = float2int8_avx(_p2, _p3);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _pp = _mm256_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                _mm256_store_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += B_hstep * 4;
            }
#else // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVX__
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + 8);
                __m256 _p2 = _mm256_loadu_ps(p0 + 16);
                __m256 _p3 = _mm256_loadu_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p1);
                __m128i _pp1 = float2int8_avx(_p2, _p3);
#else
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                __m128 _p2 = _mm_load_ps(p0 + 8);
                __m128 _p3 = _mm_load_ps(p0 + 12);
                __m128 _p4 = _mm_load_ps(p0 + 16);
                __m128 _p5 = _mm_load_ps(p0 + 20);
                __m128 _p6 = _mm_load_ps(p0 + 24);
                __m128 _p7 = _mm_load_ps(p0 + 28);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                _p2 = _mm_mul_ps(_p2, _scale);
                _p3 = _mm_mul_ps(_p3, _scale);
                _p4 = _mm_mul_ps(_p4, _scale);
                _p5 = _mm_mul_ps(_p5, _scale);
                _p6 = _mm_mul_ps(_p6, _scale);
                _p7 = _mm_mul_ps(_p7, _scale);

                __m128i _pp0 = float2int8_sse(_p0, _p1, _p2, _p3);
                __m128i _pp1 = float2int8_sse(_p4, _p5, _p6, _p7);
#endif
                __m128i _t0 = _mm_unpacklo_epi16(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi16(_pp0, _pp1);
                __m128i _t2 = _mm_unpacklo_epi16(_t0, _t1);
                __m128i _t3 = _mm_unpackhi_epi16(_t0, _t1);
                _t0 = _mm_unpacklo_epi16(_t2, _t3);
                _t1 = _mm_unpackhi_epi16(_t2, _t3);

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
        }
        if (elempack == 1)
        {
#if __AVX__
            __m256 _scale = _mm256_set1_ps(scale);
#else
            __m128 _scale = _mm_set1_ps(scale);
#endif

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + B_hstep);
                __m256 _p2 = _mm256_loadu_ps(p0 + B_hstep * 2);
                __m256 _p3 = _mm256_loadu_ps(p0 + B_hstep * 3);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);

                __m128i _tt0 = _mm_unpacklo_epi8(_pp0, _pp1);
                __m128i _tt1 = _mm_unpackhi_epi8(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi16(_tt0, _tt1);
                _pp1 = _mm_unpackhi_epi16(_tt0, _tt1);

                __m256i _pp = combine4x2_epi32(_pp0, _pp1);
#if !__AVXVNNIINT8__
                _pp = _mm256_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                _mm256_storeu_si256((__m256i*)pp, _pp);

                pp += 32;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX__
                __m256 _p0 = _mm256_loadu_ps(p0);
                __m256 _p1 = _mm256_loadu_ps(p0 + B_hstep);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);

                __m128i _pp = float2int8_avx(_p0, _p1);

                __m128i _si = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);
#else
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + 4);
                __m128 _p2 = _mm_loadu_ps(p0 + B_hstep);
                __m128 _p3 = _mm_loadu_ps(p0 + B_hstep + 4);

                __m128 _t0 = _mm_unpacklo_ps(_p0, _p2);
                __m128 _t1 = _mm_unpackhi_ps(_p0, _p2);
                __m128 _t2 = _mm_unpacklo_ps(_p1, _p3);
                __m128 _t3 = _mm_unpackhi_ps(_p1, _p3);

                _t0 = _mm_mul_ps(_t0, _scale);
                _t1 = _mm_mul_ps(_t1, _scale);
                _t2 = _mm_mul_ps(_t2, _scale);
                _t3 = _mm_mul_ps(_t3, _scale);

                __m128i _pp = float2int8_sse(_t0, _t1, _t2, _t3);
#endif

#if __AVX512F__
                _mm_store_si128((__m128i*)pp, _pp);
#else
                _mm_storeu_si128((__m128i*)pp, _pp);
#endif

                pp += 16;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
#if __AVX__
                __m256 _p = _mm256_loadu_ps(p0);

                _p = _mm256_mul_ps(_p, _scale);

                int64_t v = float2int8_avx(_p);
#else
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + 4);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);

                int64_t v = float2int8_sse(_p0, _p1);
#endif
                *(int64_t*)pp = v;

                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            __m512 _scale = _mm512_set1_ps(scale);

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose4x4_epi32(_pp0, _pp1, _pp2, _pp3);

                _pp0 = _mm_add_epi8(_pp0, _v127);
                _pp1 = _mm_add_epi8(_pp1, _v127);
                _pp2 = _mm_add_epi8(_pp2, _v127);
                _pp3 = _mm_add_epi8(_pp3, _v127);

                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);
                _mm_store_si128((__m128i*)(pp + 32), _pp2);
                _mm_store_si128((__m128i*)(pp + 48), _pp3);

                pp += 64;
                p0 += B_hstep * 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);
                __m512 _p2 = _mm512_load_ps(p0 + 32);
                __m512 _p3 = _mm512_load_ps(p0 + 48);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);
                _p2 = _mm512_mul_ps(_p2, _scale);
                _p3 = _mm512_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);
                __m128i _pp2 = float2int8_avx512(_p2);
                __m128i _pp3 = float2int8_avx512(_p3);

                transpose8x4_epi16(_pp0, _pp1, _pp2, _pp3);

                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);
                _mm_store_si128((__m128i*)(pp + 32), _pp2);
                _mm_store_si128((__m128i*)(pp + 48), _pp3);

                pp += 64;
                p0 += B_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            __m256 _scale = _mm256_set1_ps(scale);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);

                __m128i _t0 = _mm_unpacklo_epi32(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi32(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi64(_t0, _t1);
                _pp1 = _mm_unpackhi_epi64(_t0, _t1);
#if !__AVXVNNIINT8__
                _pp0 = _mm_add_epi8(_pp0, _v127);
                _pp1 = _mm_add_epi8(_pp1, _v127);
#endif // !__AVXVNNIINT8__
                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);

                pp += 32;
                p0 += B_hstep * 8;
            }
#else  // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);
                __m256 _p2 = _mm256_load_ps(p0 + 16);
                __m256 _p3 = _mm256_load_ps(p0 + 24);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);
                _p2 = _mm256_mul_ps(_p2, _scale);
                _p3 = _mm256_mul_ps(_p3, _scale);

                __m128i _pp0 = float2int8_avx(_p0, _p2);
                __m128i _pp1 = float2int8_avx(_p1, _p3);

                __m128i _t0 = _mm_unpacklo_epi16(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi16(_pp0, _pp1);
                _pp0 = _mm_unpacklo_epi32(_t0, _t1);
                _pp1 = _mm_unpackhi_epi32(_t0, _t1);

                _mm_store_si128((__m128i*)pp, _pp0);
                _mm_store_si128((__m128i*)(pp + 16), _pp1);

                pp += 32;
                p0 += B_hstep * 8;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            __m128 _scale = _mm_set1_ps(scale);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                __m128 _p2 = _mm_load_ps(p0 + 8);
                __m128 _p3 = _mm_load_ps(p0 + 12);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                _p2 = _mm_mul_ps(_p2, _scale);
                _p3 = _mm_mul_ps(_p3, _scale);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);
#if !__AVXVNNIINT8__
                _pp = _mm_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += B_hstep * 4;
            }
#else  // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                __m128 _p2 = _mm_load_ps(p0 + 8);
                __m128 _p3 = _mm_load_ps(p0 + 12);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                _p2 = _mm_mul_ps(_p2, _scale);
                _p3 = _mm_mul_ps(_p3, _scale);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);

                _pp = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pp, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                _pp = _mm_shuffle_epi32(_pp, _MM_SHUFFLE(3, 1, 2, 0));

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
        }
        if (elempack == 1)
        {
            __m128 _scale = _mm_set1_ps(scale);

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + B_hstep);
                __m128 _p2 = _mm_loadu_ps(p0 + B_hstep * 2);
                __m128 _p3 = _mm_loadu_ps(p0 + B_hstep * 3);

                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                _p2 = _mm_mul_ps(_p2, _scale);
                _p3 = _mm_mul_ps(_p3, _scale);

                __m128i _pp = float2int8_sse(_p0, _p1, _p2, _p3);

                __m128i _si = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                _pp = _mm_shuffle_epi8(_pp, _si);
#if !__AVXVNNIINT8__
                _pp = _mm_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
                _mm_storeu_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_loadu_ps(p0);
                __m128 _p1 = _mm_loadu_ps(p0 + B_hstep);
                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
                __m128 _t0 = _mm_unpacklo_ps(_p0, _p1);
                __m128 _t1 = _mm_unpackhi_ps(_p0, _p1);
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
                pp += 8;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p = _mm_loadu_ps(p0);
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            __m512 _scale = _mm512_set1_ps(scale);

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p0 = _mm512_load_ps(p0);
                __m512 _p1 = _mm512_load_ps(p0 + 16);

                _p0 = _mm512_mul_ps(_p0, _scale);
                _p1 = _mm512_mul_ps(_p1, _scale);

                __m128i _pp0 = float2int8_avx512(_p0);
                __m128i _pp1 = float2int8_avx512(_p1);

#if __AVX512VNNI__
                __m128i _t0 = _mm_unpacklo_epi32(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi32(_pp0, _pp1);

                _t0 = _mm_add_epi8(_t0, _v127);
                _t1 = _mm_add_epi8(_t1, _v127);
#else  // __AVX512VNNI__
                __m128i _t0 = _mm_unpacklo_epi16(_pp0, _pp1);
                __m128i _t1 = _mm_unpackhi_epi16(_pp0, _pp1);
#endif // __AVX512VNNI__

                _mm_store_si128((__m128i*)pp, _t0);
                _mm_store_si128((__m128i*)(pp + 16), _t1);

                pp += 32;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            __m256 _scale = _mm256_set1_ps(scale);

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p0 = _mm256_load_ps(p0);
                __m256 _p1 = _mm256_load_ps(p0 + 8);

                _p0 = _mm256_mul_ps(_p0, _scale);
                _p1 = _mm256_mul_ps(_p1, _scale);

                __m128i _pp = float2int8_avx(_p0, _p1);

                _pp = _mm_shuffle_epi32(_pp, _MM_SHUFFLE(3, 1, 2, 0));
#if __AVX512VNNI__ || __AVXVNNI__
#if !__AVXVNNIINT8__
                _pp = _mm_add_epi8(_pp, _v127);
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                _pp = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pp, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
#endif // __AVX512VNNI__ || __AVXVNNI__

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            __m128 _scale = _mm_set1_ps(scale);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_load_ps(p0);
                __m128 _p1 = _mm_load_ps(p0 + 4);
                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
#if __AVX512VNNI__ || __AVXVNNI__
                int64_t v = float2int8_sse(_p0, _p1);
                *(int64_t*)pp = v;
#if !__AVXVNNIINT8__
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
                pp[4] += 127;
                pp[5] += 127;
                pp[6] += 127;
                pp[7] += 127;
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128 _t0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                __m128 _t1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
            __m128 _scale = _mm_set1_ps(scale);
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _p1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + B_hstep)));
                __m128 _p2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + B_hstep * 2)));
                __m128 _p3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + B_hstep * 3)));
                __m128 _p01 = _mm_unpacklo_ps(_p0, _p1);
                __m128 _p23 = _mm_unpacklo_ps(_p2, _p3);
                _p0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p01), _mm_castps_pd(_p23)));
                _p1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_p01), _mm_castps_pd(_p23)));
                _p0 = _mm_mul_ps(_p0, _scale);
                _p1 = _mm_mul_ps(_p1, _scale);
#if __AVX512VNNI__ || __AVXVNNI__
                int64_t v = float2int8_sse(_p0, _p1);
                *(int64_t*)pp = v;
#if !__AVXVNNIINT8__
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
                pp[4] += 127;
                pp[5] += 127;
                pp[6] += 127;
                pp[7] += 127;
#endif // !__AVXVNNIINT8__
#else  // __AVX512VNNI__ || __AVXVNNI__
                __m128 _t0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                __m128 _t1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_p0), _mm_castps_pd(_p1)));
                int64_t v = float2int8_sse(_t0, _t1);
                *(int64_t*)pp = v;
#endif // __AVX512VNNI__ || __AVXVNNI__
                pp += 8;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _p0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _p1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(p0 + B_hstep)));
                __m128 _p = _mm_unpacklo_ps(_p0, _p1);
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
                pp += 4;
                p0 += B_hstep * 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX512VNNI__
        __m128i _v127 = _mm_set1_epi8(127);
#endif // __AVX512VNNI__

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            __m512 _scale = _mm512_set1_ps(scale);

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _p = _mm512_load_ps(p0);

                _p = _mm512_mul_ps(_p, _scale);

                __m128i _pp = float2int8_avx512(_p);

#if __AVX512VNNI__
                _pp = _mm_add_epi8(_pp, _v127);
#endif // __AVX512VNNI__

                _mm_store_si128((__m128i*)pp, _pp);

                pp += 16;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            __m256 _scale = _mm256_set1_ps(scale);

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _p = _mm256_load_ps(p0);
                _p = _mm256_mul_ps(_p, _scale);
                int64_t v = float2int8_avx(_p);
                *(int64_t*)pp = v;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
                pp[4] += 127;
                pp[5] += 127;
                pp[6] += 127;
                pp[7] += 127;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            __m128 _scale = _mm_set1_ps(scale);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _p = _mm_load_ps(p0);
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
            __m128 _scale = _mm_set1_ps(scale);
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVX2__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(B_hstep));

                __m128 _p = _mm_i32gather_ps(p0, _vindex, sizeof(float));
#else
                __m128 _p = _mm_setr_ps(p0[0], p0[B_hstep], p0[B_hstep * 2], p0[B_hstep * 3]);
#endif
                _p = _mm_mul_ps(_p, _scale);
                int32_t v = float2int8_sse(_p);
                *(int32_t*)pp = v;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
                pp += 4;
                p0 += B_hstep * 4;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void unpack_output_tile_int32_to_fp32(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        unpack_output_tile_int32_to_fp32_avx2(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta, output_transpose);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const int c_hstep = C.dims == 3 ? (int)C.cstep : C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    // NCNN_LOGE("unpack_output_tile_int32_to_fp32  %d %d %d %d  %d  %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C, c_elempack, output_transpose);

    const int* pp = topT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m512 _descale = _mm512_load_ps((const float*)descales + i + ii);

        __m512 _c0 = _mm512_set1_ps(0.f);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm512_set1_ps(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm512_loadu_ps(pC);
                _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 48)));
            __m512 _f4 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 64)));
            __m512 _f5 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 80)));
            __m512 _f6 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 96)));
            __m512 _f7 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 112)));
            __m512 _f8 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128)));
            __m512 _f9 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 16)));
            __m512 _fa = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 32)));
            __m512 _fb = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 48)));
            __m512 _fc = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 64)));
            __m512 _fd = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 80)));
            __m512 _fe = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 96)));
            __m512 _ff = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 112)));
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
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
                _f9 = _mm512_permute_ps(_f9, _MM_SHUFFLE(2, 1, 0, 3));
                _fb = _mm512_permute_ps(_fb, _MM_SHUFFLE(2, 1, 0, 3));
                _fd = _mm512_permute_ps(_fd, _MM_SHUFFLE(2, 1, 0, 3));
                _ff = _mm512_permute_ps(_ff, _MM_SHUFFLE(2, 1, 0, 3));

                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_f4, _f7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_f4, _f7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_f6, _f5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f5);
                __m512 _tmp8 = _mm512_unpacklo_ps(_f8, _fb);
                __m512 _tmp9 = _mm512_unpackhi_ps(_f8, _fb);
                __m512 _tmpa = _mm512_unpacklo_ps(_fa, _f9);
                __m512 _tmpb = _mm512_unpackhi_ps(_fa, _f9);
                __m512 _tmpc = _mm512_unpacklo_ps(_fc, _ff);
                __m512 _tmpd = _mm512_unpackhi_ps(_fc, _ff);
                __m512 _tmpe = _mm512_unpacklo_ps(_fe, _fd);
                __m512 _tmpf = _mm512_unpackhi_ps(_fe, _fd);

                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp8), _mm512_castps_pd(_tmpa)));
                _f9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp8), _mm512_castps_pd(_tmpa)));
                _fa = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpb), _mm512_castps_pd(_tmp9)));
                _fb = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpb), _mm512_castps_pd(_tmp9)));
                _fc = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpc), _mm512_castps_pd(_tmpe)));
                _fd = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpc), _mm512_castps_pd(_tmpe)));
                _fe = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpf), _mm512_castps_pd(_tmpd)));
                _ff = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpf), _mm512_castps_pd(_tmpd)));

                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
                _f9 = _mm512_permute_ps(_f9, _MM_SHUFFLE(2, 1, 0, 3));
                _fb = _mm512_permute_ps(_fb, _MM_SHUFFLE(2, 1, 0, 3));
                _fd = _mm512_permute_ps(_fd, _MM_SHUFFLE(2, 1, 0, 3));
                _ff = _mm512_permute_ps(_ff, _MM_SHUFFLE(2, 1, 0, 3));

                _tmp0 = _mm512_shuffle_f32x4(_f0, _f8, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp1 = _mm512_shuffle_f32x4(_f1, _f9, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp2 = _mm512_shuffle_f32x4(_f2, _fa, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp3 = _mm512_shuffle_f32x4(_f3, _fb, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp4 = _mm512_shuffle_f32x4(_f8, _f0, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp5 = _mm512_shuffle_f32x4(_f9, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp6 = _mm512_shuffle_f32x4(_fa, _f2, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp7 = _mm512_shuffle_f32x4(_fb, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp8 = _mm512_shuffle_f32x4(_f4, _fc, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp9 = _mm512_shuffle_f32x4(_f5, _fd, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpa = _mm512_shuffle_f32x4(_f6, _fe, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpb = _mm512_shuffle_f32x4(_f7, _ff, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpc = _mm512_shuffle_f32x4(_fc, _f4, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpd = _mm512_shuffle_f32x4(_fd, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpe = _mm512_shuffle_f32x4(_fe, _f6, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpf = _mm512_shuffle_f32x4(_ff, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 1, 2, 0));
                _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 1, 2, 0));
                _f2 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 1, 2, 0));
                _f3 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 1, 2, 0));
                _f4 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 1, 2, 0));
                _f5 = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 1, 2, 0));
                _f6 = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 1, 2, 0));
                _f7 = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 1, 2, 0));
                _f8 = _mm512_shuffle_f32x4(_tmp8, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                _f9 = _mm512_shuffle_f32x4(_tmp9, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                _fa = _mm512_shuffle_f32x4(_tmpa, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                _fb = _mm512_shuffle_f32x4(_tmpb, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                _fc = _mm512_shuffle_f32x4(_tmpc, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                _fd = _mm512_shuffle_f32x4(_tmpd, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                _fe = _mm512_shuffle_f32x4(_tmpe, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                _ff = _mm512_shuffle_f32x4(_tmpf, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));
            }

            _f0 = _mm512_mul_ps(_f0, _descale);
            _f1 = _mm512_mul_ps(_f1, _descale);
            _f2 = _mm512_mul_ps(_f2, _descale);
            _f3 = _mm512_mul_ps(_f3, _descale);
            _f4 = _mm512_mul_ps(_f4, _descale);
            _f5 = _mm512_mul_ps(_f5, _descale);
            _f6 = _mm512_mul_ps(_f6, _descale);
            _f7 = _mm512_mul_ps(_f7, _descale);
            _f8 = _mm512_mul_ps(_f8, _descale);
            _f9 = _mm512_mul_ps(_f9, _descale);
            _fa = _mm512_mul_ps(_fa, _descale);
            _fb = _mm512_mul_ps(_fb, _descale);
            _fc = _mm512_mul_ps(_fc, _descale);
            _fd = _mm512_mul_ps(_fd, _descale);
            _fe = _mm512_mul_ps(_fe, _descale);
            _ff = _mm512_mul_ps(_ff, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c0);
                    _f6 = _mm512_add_ps(_f6, _c0);
                    _f7 = _mm512_add_ps(_f7, _c0);
                    _f8 = _mm512_add_ps(_f8, _c0);
                    _f9 = _mm512_add_ps(_f9, _c0);
                    _fa = _mm512_add_ps(_fa, _c0);
                    _fb = _mm512_add_ps(_fb, _c0);
                    _fc = _mm512_add_ps(_fc, _c0);
                    _fd = _mm512_add_ps(_fd, _c0);
                    _fe = _mm512_add_ps(_fe, _c0);
                    _ff = _mm512_add_ps(_ff, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c0);
                    _f6 = _mm512_add_ps(_f6, _c0);
                    _f7 = _mm512_add_ps(_f7, _c0);
                    _f8 = _mm512_add_ps(_f8, _c0);
                    _f9 = _mm512_add_ps(_f9, _c0);
                    _fa = _mm512_add_ps(_fa, _c0);
                    _fb = _mm512_add_ps(_fb, _c0);
                    _fc = _mm512_add_ps(_fc, _c0);
                    _fd = _mm512_add_ps(_fd, _c0);
                    _fe = _mm512_add_ps(_fe, _c0);
                    _ff = _mm512_add_ps(_ff, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    __m512 _c4;
                    __m512 _c5;
                    __m512 _c6;
                    __m512 _c7;
                    __m512 _c8;
                    __m512 _c9;
                    __m512 _ca;
                    __m512 _cb;
                    __m512 _cc;
                    __m512 _cd;
                    __m512 _ce;
                    __m512 _cf;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        _c4 = _mm512_loadu_ps(pC + 64);
                        _c5 = _mm512_loadu_ps(pC + 80);
                        _c6 = _mm512_loadu_ps(pC + 96);
                        _c7 = _mm512_loadu_ps(pC + 112);
                        _c8 = _mm512_loadu_ps(pC + 128);
                        _c9 = _mm512_loadu_ps(pC + 128 + 16);
                        _ca = _mm512_loadu_ps(pC + 128 + 32);
                        _cb = _mm512_loadu_ps(pC + 128 + 48);
                        _cc = _mm512_loadu_ps(pC + 128 + 64);
                        _cd = _mm512_loadu_ps(pC + 128 + 80);
                        _ce = _mm512_loadu_ps(pC + 128 + 96);
                        _cf = _mm512_loadu_ps(pC + 128 + 112);
                        pC += 256;
                    }
                    else if (c_elempack == 8)
                    {
                        __m512 _tmp0 = _mm512_loadu_ps(pC);
                        __m512 _tmp1 = _mm512_loadu_ps(pC + 16);
                        __m512 _tmp2 = _mm512_loadu_ps(pC + 32);
                        __m512 _tmp3 = _mm512_loadu_ps(pC + 48);
                        __m512 _tmp4 = _mm512_loadu_ps(pC + 64);
                        __m512 _tmp5 = _mm512_loadu_ps(pC + 80);
                        __m512 _tmp6 = _mm512_loadu_ps(pC + 96);
                        __m512 _tmp7 = _mm512_loadu_ps(pC + 112);
                        __m512 _tmp8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _tmp9 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        __m512 _tmpa = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        __m512 _tmpb = _mm512_loadu_ps(pC + c_hstep * 8 + 48);
                        __m512 _tmpc = _mm512_loadu_ps(pC + c_hstep * 8 + 64);
                        __m512 _tmpd = _mm512_loadu_ps(pC + c_hstep * 8 + 80);
                        __m512 _tmpe = _mm512_loadu_ps(pC + c_hstep * 8 + 96);
                        __m512 _tmpf = _mm512_loadu_ps(pC + c_hstep * 8 + 112);

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 2, 3, 2));
                        _c4 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
                        _c6 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
                        _c8 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(1, 0, 1, 0));
                        _c9 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 2, 3, 2));
                        _ca = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(1, 0, 1, 0));
                        _cb = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 2, 3, 2));
                        _cc = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
                        _cd = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
                        _ce = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
                        _cf = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

                        pC += 128;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 4 + 32);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 4 + 48);
                        _c8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c9 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _ca = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        _cb = _mm512_loadu_ps(pC + c_hstep * 8 + 48);
                        _cc = _mm512_loadu_ps(pC + c_hstep * 12);
                        _cd = _mm512_loadu_ps(pC + c_hstep * 12 + 16);
                        _ce = _mm512_loadu_ps(pC + c_hstep * 12 + 32);
                        _cf = _mm512_loadu_ps(pC + c_hstep * 12 + 48);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c4, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c8, _cc, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c4, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c8, _cc, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c1, _c5, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c9, _cd, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c1, _c5, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c9, _cd, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp8 = _mm512_shuffle_f32x4(_c2, _c6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp9 = _mm512_shuffle_f32x4(_ca, _ce, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpa = _mm512_shuffle_f32x4(_c2, _c6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpb = _mm512_shuffle_f32x4(_ca, _ce, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpc = _mm512_shuffle_f32x4(_c3, _c7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpd = _mm512_shuffle_f32x4(_cb, _cf, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpe = _mm512_shuffle_f32x4(_c3, _c7, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpf = _mm512_shuffle_f32x4(_cb, _cf, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                        _c8 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                        _c9 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                        _ca = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                        _cb = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
                        _cc = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                        _cd = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                        _ce = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
                        _cf = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                        pC += 64;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + c_hstep);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 3);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 5);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 6);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 7);
                        _c8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c9 = _mm512_loadu_ps(pC + c_hstep * 9);
                        _ca = _mm512_loadu_ps(pC + c_hstep * 10);
                        _cb = _mm512_loadu_ps(pC + c_hstep * 11);
                        _cc = _mm512_loadu_ps(pC + c_hstep * 12);
                        _cd = _mm512_loadu_ps(pC + c_hstep * 13);
                        _ce = _mm512_loadu_ps(pC + c_hstep * 14);
                        _cf = _mm512_loadu_ps(pC + c_hstep * 15);
                        transpose16x16_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _ca, _cb, _cc, _cd, _ce, _cf);
                        pC += 16;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                        _f2 = _mm512_add_ps(_f2, _c2);
                        _f3 = _mm512_add_ps(_f3, _c3);
                        _f4 = _mm512_add_ps(_f4, _c4);
                        _f5 = _mm512_add_ps(_f5, _c5);
                        _f6 = _mm512_add_ps(_f6, _c6);
                        _f7 = _mm512_add_ps(_f7, _c7);
                        _f8 = _mm512_add_ps(_f8, _c8);
                        _f9 = _mm512_add_ps(_f9, _c9);
                        _fa = _mm512_add_ps(_fa, _ca);
                        _fb = _mm512_add_ps(_fb, _cb);
                        _fc = _mm512_add_ps(_fc, _cc);
                        _fd = _mm512_add_ps(_fd, _cd);
                        _fe = _mm512_add_ps(_fe, _ce);
                        _ff = _mm512_add_ps(_ff, _cf);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7, _beta, _f7);
                        _f8 = _mm512_fmadd_ps(_c8, _beta, _f8);
                        _f9 = _mm512_fmadd_ps(_c9, _beta, _f9);
                        _fa = _mm512_fmadd_ps(_ca, _beta, _fa);
                        _fb = _mm512_fmadd_ps(_cb, _beta, _fb);
                        _fc = _mm512_fmadd_ps(_cc, _beta, _fc);
                        _fd = _mm512_fmadd_ps(_cd, _beta, _fd);
                        _fe = _mm512_fmadd_ps(_ce, _beta, _fe);
                        _ff = _mm512_fmadd_ps(_cf, _beta, _ff);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
                    _f3 = _mm512_add_ps(_f3, _c3);

                    _c0 = _mm512_set1_ps(pC[4] * beta);
                    _c1 = _mm512_set1_ps(pC[5] * beta);
                    _c2 = _mm512_set1_ps(pC[6] * beta);
                    _c3 = _mm512_set1_ps(pC[7] * beta);

                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c1);
                    _f6 = _mm512_add_ps(_f6, _c2);
                    _f7 = _mm512_add_ps(_f7, _c3);

                    _c0 = _mm512_set1_ps(pC[8] * beta);
                    _c1 = _mm512_set1_ps(pC[9] * beta);
                    _c2 = _mm512_set1_ps(pC[10] * beta);
                    _c3 = _mm512_set1_ps(pC[11] * beta);

                    _f8 = _mm512_add_ps(_f8, _c0);
                    _f9 = _mm512_add_ps(_f9, _c1);
                    _fa = _mm512_add_ps(_fa, _c2);
                    _fb = _mm512_add_ps(_fb, _c3);

                    _c0 = _mm512_set1_ps(pC[12] * beta);
                    _c1 = _mm512_set1_ps(pC[13] * beta);
                    _c2 = _mm512_set1_ps(pC[14] * beta);
                    _c3 = _mm512_set1_ps(pC[15] * beta);

                    _fc = _mm512_add_ps(_fc, _c0);
                    _fd = _mm512_add_ps(_fd, _c1);
                    _fe = _mm512_add_ps(_fe, _c2);
                    _ff = _mm512_add_ps(_ff, _c3);
                    pC += 16;
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
                _f8 = _mm512_mul_ps(_f8, _alpha);
                _f9 = _mm512_mul_ps(_f9, _alpha);
                _fa = _mm512_mul_ps(_fa, _alpha);
                _fb = _mm512_mul_ps(_fb, _alpha);
                _fc = _mm512_mul_ps(_fc, _alpha);
                _fd = _mm512_mul_ps(_fd, _alpha);
                _fe = _mm512_mul_ps(_fe, _alpha);
                _ff = _mm512_mul_ps(_ff, _alpha);
            }

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    transpose16x16_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 16 * 2, _f2);
                    _mm512_storeu_ps(p0 + 16 * 3, _f3);
                    _mm512_storeu_ps(p0 + 16 * 4, _f4);
                    _mm512_storeu_ps(p0 + 16 * 5, _f5);
                    _mm512_storeu_ps(p0 + 16 * 6, _f6);
                    _mm512_storeu_ps(p0 + 16 * 7, _f7);
                    _mm512_storeu_ps(p0 + 16 * 8, _f8);
                    _mm512_storeu_ps(p0 + 16 * 9, _f9);
                    _mm512_storeu_ps(p0 + 16 * 10, _fa);
                    _mm512_storeu_ps(p0 + 16 * 11, _fb);
                    _mm512_storeu_ps(p0 + 16 * 12, _fc);
                    _mm512_storeu_ps(p0 + 16 * 13, _fd);
                    _mm512_storeu_ps(p0 + 16 * 14, _fe);
                    _mm512_storeu_ps(p0 + 16 * 15, _ff);
                }
                if (out_elempack == 8)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    transpose16x8_ps(_f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 16 * 2, _f2);
                    _mm512_storeu_ps(p0 + 16 * 3, _f3);
                    _mm512_storeu_ps(p0 + 16 * 4, _f4);
                    _mm512_storeu_ps(p0 + 16 * 5, _f5);
                    _mm512_storeu_ps(p0 + 16 * 6, _f6);
                    _mm512_storeu_ps(p0 + 16 * 7, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 2, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 3, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 4, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 5, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 6, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 7, _ff);
                }
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    transpose16x4_ps(_f4, _f5, _f6, _f7);
                    transpose16x4_ps(_f8, _f9, _fa, _fb);
                    transpose16x4_ps(_fc, _fd, _fe, _ff);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 32, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 48, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 32, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 48, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 16, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 32, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 48, _ff);
                }
                if (out_elempack == 1)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 7, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 9, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 10, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 11, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 13, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 14, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 15, _ff);
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(p0, _f0);
                    _mm512_store_ps(p0 + 16, _f1);
                    _mm512_store_ps(p0 + 32, _f2);
                    _mm512_store_ps(p0 + 48, _f3);
                    _mm512_store_ps(p0 + 64, _f4);
                    _mm512_store_ps(p0 + 80, _f5);
                    _mm512_store_ps(p0 + 96, _f6);
                    _mm512_store_ps(p0 + 112, _f7);
                    _mm512_store_ps(p0 + 128, _f8);
                    _mm512_store_ps(p0 + 128 + 16, _f9);
                    _mm512_store_ps(p0 + 128 + 32, _fa);
                    _mm512_store_ps(p0 + 128 + 48, _fb);
                    _mm512_store_ps(p0 + 128 + 64, _fc);
                    _mm512_store_ps(p0 + 128 + 80, _fd);
                    _mm512_store_ps(p0 + 128 + 96, _fe);
                    _mm512_store_ps(p0 + 128 + 112, _ff);
                    p0 += 256;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_store_ps(p0 + 8, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_store_ps(p0 + 16, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_store_ps(p0 + 24, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_store_ps(p0 + 32, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_store_ps(p0 + 40, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_store_ps(p0 + 48, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_store_ps(p0 + 56, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_store_ps(p0 + 64, _mm512_extractf32x8_ps(_f8, 0));
                    _mm256_store_ps(p0 + 64 + 8, _mm512_extractf32x8_ps(_f9, 0));
                    _mm256_store_ps(p0 + 64 + 16, _mm512_extractf32x8_ps(_fa, 0));
                    _mm256_store_ps(p0 + 64 + 24, _mm512_extractf32x8_ps(_fb, 0));
                    _mm256_store_ps(p0 + 64 + 32, _mm512_extractf32x8_ps(_fc, 0));
                    _mm256_store_ps(p0 + 64 + 40, _mm512_extractf32x8_ps(_fd, 0));
                    _mm256_store_ps(p0 + 64 + 48, _mm512_extractf32x8_ps(_fe, 0));
                    _mm256_store_ps(p0 + 64 + 56, _mm512_extractf32x8_ps(_ff, 0));
                    _mm256_store_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 8, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 16, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 24, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 32, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 40, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 48, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 56, _mm512_extractf32x8_ps(_f7, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 64, _mm512_extractf32x8_ps(_f8, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 64 + 8, _mm512_extractf32x8_ps(_f9, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 64 + 16, _mm512_extractf32x8_ps(_fa, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 64 + 24, _mm512_extractf32x8_ps(_fb, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 64 + 32, _mm512_extractf32x8_ps(_fc, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 64 + 40, _mm512_extractf32x8_ps(_fd, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 64 + 48, _mm512_extractf32x8_ps(_fe, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 64 + 56, _mm512_extractf32x8_ps(_ff, 1));
                    p0 += 128;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_f8, _f9, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_fa, _fb, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_fc, _fd, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_fe, _ff, _MM_SHUFFLE(2, 0, 2, 0));

                    __m512 _tmp8 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpc = _mm512_shuffle_f32x4(_f8, _f9, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpd = _mm512_shuffle_f32x4(_fa, _fb, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpe = _mm512_shuffle_f32x4(_fc, _fd, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpf = _mm512_shuffle_f32x4(_fe, _ff, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f4 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                    _f5 = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                    _f6 = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _f7 = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));

                    _f8 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f9 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _fa = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _fb = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _fc = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                    _fd = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
                    _fe = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                    _ff = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 32, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 48, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 32, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 48, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 16, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 32, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 48, _ff);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    transpose16x16_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 7, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 9, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 10, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 11, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 13, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 14, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 15, _ff);
                    p0 += 16;
                }
            }
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 48)));
            __m512 _f4 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 64)));
            __m512 _f5 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 80)));
            __m512 _f6 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 96)));
            __m512 _f7 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 112)));
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
            }

            _f0 = _mm512_mul_ps(_f0, _descale);
            _f1 = _mm512_mul_ps(_f1, _descale);
            _f2 = _mm512_mul_ps(_f2, _descale);
            _f3 = _mm512_mul_ps(_f3, _descale);
            _f4 = _mm512_mul_ps(_f4, _descale);
            _f5 = _mm512_mul_ps(_f5, _descale);
            _f6 = _mm512_mul_ps(_f6, _descale);
            _f7 = _mm512_mul_ps(_f7, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c0);
                    _f6 = _mm512_add_ps(_f6, _c0);
                    _f7 = _mm512_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c0);
                    _f6 = _mm512_add_ps(_f6, _c0);
                    _f7 = _mm512_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    __m512 _c4;
                    __m512 _c5;
                    __m512 _c6;
                    __m512 _c7;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        _c4 = _mm512_loadu_ps(pC + 64);
                        _c5 = _mm512_loadu_ps(pC + 80);
                        _c6 = _mm512_loadu_ps(pC + 96);
                        _c7 = _mm512_loadu_ps(pC + 112);
                        pC += 128;
                    }
                    else if (c_elempack == 8)
                    {
                        __m512 _tmp0 = _mm512_loadu_ps(pC);
                        __m512 _tmp1 = _mm512_loadu_ps(pC + 16);
                        __m512 _tmp2 = _mm512_loadu_ps(pC + 32);
                        __m512 _tmp3 = _mm512_loadu_ps(pC + 48);
                        __m512 _tmp4 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _tmp5 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        __m512 _tmp6 = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        __m512 _tmp7 = _mm512_loadu_ps(pC + c_hstep * 8 + 48);

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(3, 2, 3, 2));
                        _c4 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                        _c6 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));

                        pC += 64;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 12);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 12 + 16);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c4, _c6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c4, _c6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c1, _c3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c5, _c7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c1, _c3, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c5, _c7, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep);
                        __m256 _cc2 = _mm256_loadu_ps(pC + c_hstep * 2);
                        __m256 _cc3 = _mm256_loadu_ps(pC + c_hstep * 3);
                        __m256 _cc4 = _mm256_loadu_ps(pC + c_hstep * 4);
                        __m256 _cc5 = _mm256_loadu_ps(pC + c_hstep * 5);
                        __m256 _cc6 = _mm256_loadu_ps(pC + c_hstep * 6);
                        __m256 _cc7 = _mm256_loadu_ps(pC + c_hstep * 7);
                        __m256 _cc8 = _mm256_loadu_ps(pC + c_hstep * 8);
                        __m256 _cc9 = _mm256_loadu_ps(pC + c_hstep * 9);
                        __m256 _cca = _mm256_loadu_ps(pC + c_hstep * 10);
                        __m256 _ccb = _mm256_loadu_ps(pC + c_hstep * 11);
                        __m256 _ccc = _mm256_loadu_ps(pC + c_hstep * 12);
                        __m256 _ccd = _mm256_loadu_ps(pC + c_hstep * 13);
                        __m256 _cce = _mm256_loadu_ps(pC + c_hstep * 14);
                        __m256 _ccf = _mm256_loadu_ps(pC + c_hstep * 15);
                        transpose8x8_ps(_cc0, _cc1, _cc2, _cc3, _cc4, _cc5, _cc6, _cc7);
                        transpose8x8_ps(_cc8, _cc9, _cca, _ccb, _ccc, _ccd, _cce, _ccf);
                        _c0 = combine8x2_ps(_cc0, _cc8);
                        _c1 = combine8x2_ps(_cc1, _cc9);
                        _c2 = combine8x2_ps(_cc2, _cca);
                        _c3 = combine8x2_ps(_cc3, _ccb);
                        _c4 = combine8x2_ps(_cc4, _ccc);
                        _c5 = combine8x2_ps(_cc5, _ccd);
                        _c6 = combine8x2_ps(_cc6, _cce);
                        _c7 = combine8x2_ps(_cc7, _ccf);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                        _f2 = _mm512_add_ps(_f2, _c2);
                        _f3 = _mm512_add_ps(_f3, _c3);
                        _f4 = _mm512_add_ps(_f4, _c4);
                        _f5 = _mm512_add_ps(_f5, _c5);
                        _f6 = _mm512_add_ps(_f6, _c6);
                        _f7 = _mm512_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
                    _f3 = _mm512_add_ps(_f3, _c3);

                    _c0 = _mm512_set1_ps(pC[4] * beta);
                    _c1 = _mm512_set1_ps(pC[5] * beta);
                    _c2 = _mm512_set1_ps(pC[6] * beta);
                    _c3 = _mm512_set1_ps(pC[7] * beta);

                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c1);
                    _f6 = _mm512_add_ps(_f6, _c2);
                    _f7 = _mm512_add_ps(_f7, _c3);
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

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + 64, _f4);
                    _mm512_storeu_ps(p0 + 80, _f5);
                    _mm512_storeu_ps(p0 + 96, _f6);
                    _mm512_storeu_ps(p0 + 112, _f7);
                }
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    transpose16x4_ps(_f4, _f5, _f6, _f7);
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 32, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 48, _f7);
                }
                if (out_elempack == 1)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 7, _f7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(p0, _f0);
                    _mm512_store_ps(p0 + 16, _f1);
                    _mm512_store_ps(p0 + 32, _f2);
                    _mm512_store_ps(p0 + 48, _f3);
                    _mm512_store_ps(p0 + 64, _f4);
                    _mm512_store_ps(p0 + 80, _f5);
                    _mm512_store_ps(p0 + 96, _f6);
                    _mm512_store_ps(p0 + 112, _f7);
                    p0 += 128;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_store_ps(p0 + 8, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_store_ps(p0 + 16, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_store_ps(p0 + 24, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_store_ps(p0 + 32, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_store_ps(p0 + 40, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_store_ps(p0 + 48, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_store_ps(p0 + 56, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_store_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 8, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 16, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 24, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 32, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 40, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 48, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 56, _mm512_extractf32x8_ps(_f7, 1));
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _f6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _f7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 16, _f7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + out_hstep, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 2, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 6, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 10, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 14, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x8_ps(_f7, 1));
                    p0 += 8;
                }
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 48)));
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
            }

            _f0 = _mm512_mul_ps(_f0, _descale);
            _f1 = _mm512_mul_ps(_f1, _descale);
            _f2 = _mm512_mul_ps(_f2, _descale);
            _f3 = _mm512_mul_ps(_f3, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        pC += 64;
                    }
                    else if (c_elempack == 8)
                    {
                        __m512 _cc0 = _mm512_loadu_ps(pC);
                        __m512 _cc1 = _mm512_loadu_ps(pC + 16);
                        __m512 _cc2 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _cc3 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _c0 = _mm512_shuffle_f32x4(_cc0, _cc2, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_cc0, _cc2, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_cc1, _cc3, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_cc1, _cc3, _MM_SHUFFLE(3, 2, 3, 2));
                        pC += 32;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 12);
                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c1, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c2, _c3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c1, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c2, _c3, _MM_SHUFFLE(3, 2, 3, 2));
                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 2);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 3);
                        __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 5);
                        __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 6);
                        __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 7);
                        __m128 _cc8 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc9 = _mm_loadu_ps(pC + c_hstep * 9);
                        __m128 _cca = _mm_loadu_ps(pC + c_hstep * 10);
                        __m128 _ccb = _mm_loadu_ps(pC + c_hstep * 11);
                        __m128 _ccc = _mm_loadu_ps(pC + c_hstep * 12);
                        __m128 _ccd = _mm_loadu_ps(pC + c_hstep * 13);
                        __m128 _cce = _mm_loadu_ps(pC + c_hstep * 14);
                        __m128 _ccf = _mm_loadu_ps(pC + c_hstep * 15);
                        _MM_TRANSPOSE4_PS(_cc0, _cc1, _cc2, _cc3);
                        _MM_TRANSPOSE4_PS(_cc4, _cc5, _cc6, _cc7);
                        _MM_TRANSPOSE4_PS(_cc8, _cc9, _cca, _ccb);
                        _MM_TRANSPOSE4_PS(_ccc, _ccd, _cce, _ccf);

                        _c0 = combine4x4_ps(_cc0, _cc4, _cc8, _ccc);
                        _c1 = combine4x4_ps(_cc1, _cc5, _cc9, _ccd);
                        _c2 = combine4x4_ps(_cc2, _cc6, _cca, _cce);
                        _c3 = combine4x4_ps(_cc3, _cc7, _ccb, _ccf);

                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                        _f2 = _mm512_add_ps(_f2, _c2);
                        _f3 = _mm512_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
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

            if (output_transpose)
            {
#if !(defined(__x86_64__) || defined(_M_X64))
#if __AVX__
#if __AVX512F__
                if (out_elempack == 16)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    const int jj_m16 = jj % 16;
                    float* p1 = p0 - out_hstep * jj_m16 + jj_m16;
                    _mm_storeu_ps(p1, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p1 + 16, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p1 + 32, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p1 + 48, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p1 + 64, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p1 + 80, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p1 + 96, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p1 + 112, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_storeu_ps(p1 + 128, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_storeu_ps(p1 + 144, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_storeu_ps(p1 + 160, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_storeu_ps(p1 + 176, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_storeu_ps(p1 + 192, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_storeu_ps(p1 + 208, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_storeu_ps(p1 + 224, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_storeu_ps(p1 + 240, _mm512_extractf32x4_ps(_f3, 3));
                }
#endif // __AVX512F__
                if (out_elempack == 8)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    const int jj_m8 = jj % 8;
                    float* p1 = p0 - out_hstep * jj_m8 + jj_m8;
                    _mm_storeu_ps(p1, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p1 + 8, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p1 + 16, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p1 + 24, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p1 + 32, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p1 + 40, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p1 + 48, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p1 + 56, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_storeu_ps(p1 + 64, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_storeu_ps(p1 + 72, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_storeu_ps(p1 + 80, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_storeu_ps(p1 + 88, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_storeu_ps(p1 + 96, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_storeu_ps(p1 + 104, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_storeu_ps(p1 + 112, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_storeu_ps(p1 + 120, _mm512_extractf32x4_ps(_f3, 3));
                }
#endif // __AVX__
#endif // !(defined(__x86_64__) || defined(_M_X64))
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                }
                if (out_elempack == 1)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(p0, _f0);
                    _mm512_store_ps(p0 + 16, _f1);
                    _mm512_store_ps(p0 + 32, _f2);
                    _mm512_store_ps(p0 + 48, _f3);
                    p0 += 64;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(p0, _tmp0);
                    _mm512_storeu_ps(p0 + 16, _tmp1);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _tmp2);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _tmp3);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _f3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
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
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            pp += 32;

            // from
            //      00 11 20 31 40 51 60 71 80 91 a0 b1 c0 d1 e0 f1
            //      01 10 21 30 41 50 61 70 81 90 a1 b0 c1 d0 e1 f0
            // to
            //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
            //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
            {
                __m512 _tmp0 = _mm512_permute_ps(_f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m512 _tmp1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm512_unpacklo_ps(_tmp0, _tmp1);
                _f1 = _mm512_unpackhi_ps(_tmp0, _tmp1);
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm512_mul_ps(_f0, _descale);
            _f1 = _mm512_mul_ps(_f1, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        pC += 32;
                    }
                    else if (c_elempack == 8)
                    {
                        __m512 _cc0 = _mm512_loadu_ps(pC);
                        __m512 _cc1 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c0 = _mm512_shuffle_f32x4(_cc0, _cc1, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_cc0, _cc1, _MM_SHUFFLE(3, 2, 3, 2));
                        pC += 16;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + 4);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 4 + 4);
                        __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 8 + 4);
                        __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 12);
                        __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 12 + 4);
                        _c0 = combine4x4_ps(_cc0, _cc2, _cc4, _cc6);
                        _c1 = combine4x4_ps(_cc1, _cc3, _cc5, _cc7);
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
                        __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(c_hstep));
                        _c0 = _mm512_i32gather_ps(_vindex, pC, sizeof(float));
                        _c1 = _mm512_i32gather_ps(_vindex, pC + 1, sizeof(float));
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    _f0 = _mm512_add_ps(_f0, _c0);
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

            if (output_transpose)
            {
                _mm512_storeu_ps(p0, _f0);
                _mm512_storeu_ps(p0 + out_hstep, _f1);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(p0, _f0);
                    _mm512_store_ps(p0 + 16, _f1);
                    p0 += 32;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    _mm512_storeu_ps(p0, _tmp0);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _tmp1);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_store_ps(p0 + 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_store_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_store_ps(p0 + out_hstep * 4 + 4, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_store_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_store_ps(p0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_store_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_store_ps(p0 + out_hstep * 12 + 4, _mm512_extractf32x4_ps(_f1, 3));
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    _mm512_i32scatter_ps(p0 + 1, _vindex, _f1, sizeof(float));
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            pp += 16;

            _f0 = _mm512_mul_ps(_f0, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        pC += 16;
                    }
                    else if (c_elempack == 8)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep * 8);
                        _c0 = combine8x2_ps(_cc0, _cc1);
                        pC += 8;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 12);
                        _c0 = combine4x4_ps(_cc0, _cc1, _cc2, _cc3);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
                        __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(c_hstep));
                        _c0 = _mm512_i32gather_ps(_vindex, pC, sizeof(float));
                        pC += 1;
                    }
                    _f0 = _mm512_fmadd_ps(_c0, _mm512_set1_ps(beta), _f0);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    _f0 = _mm512_add_ps(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = _mm512_mul_ps(_f0, _mm512_set1_ps(alpha));

            if (output_transpose)
            {
                _mm512_storeu_ps(p0, _f0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(p0, _f0);
                    p0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_store_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_store_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_store_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_store_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
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
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m256 _descale = _mm256_load_ps((const float*)descales + i + ii);
#if __AVX512F__
        __m512 _descale_avx512 = _mm512_broadcast_f32x8(_descale);
#endif

        __m256 _c0 = _mm256_set1_ps(0.f);
#if __AVX512F__
        __m512 _c0_avx512 = _mm512_set1_ps(0.f);
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm256_set1_ps(pC[0] * beta);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm256_loadu_ps(pC);
                _c0 = _mm256_mul_ps(_c0, _mm256_set1_ps(beta));
#if __AVX512F__
                _c0_avx512 = _mm512_broadcast_f32x8(_c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 48)));
            __m512 _f4 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 64)));
            __m512 _f5 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 80)));
            __m512 _f6 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 96)));
            __m512 _f7 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 112)));
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
                _tmp1 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp2 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp4 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp5 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp7 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));

                _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _f2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _f3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(1, 3, 1, 3));
                _f5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(1, 3, 1, 3));
                _f6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _f7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
            }

            _f0 = _mm512_mul_ps(_f0, _descale_avx512);
            _f1 = _mm512_mul_ps(_f1, _descale_avx512);
            _f2 = _mm512_mul_ps(_f2, _descale_avx512);
            _f3 = _mm512_mul_ps(_f3, _descale_avx512);
            _f4 = _mm512_mul_ps(_f4, _descale_avx512);
            _f5 = _mm512_mul_ps(_f5, _descale_avx512);
            _f6 = _mm512_mul_ps(_f6, _descale_avx512);
            _f7 = _mm512_mul_ps(_f7, _descale_avx512);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                    _f4 = _mm512_add_ps(_f4, _c0_avx512);
                    _f5 = _mm512_add_ps(_f5, _c0_avx512);
                    _f6 = _mm512_add_ps(_f6, _c0_avx512);
                    _f7 = _mm512_add_ps(_f7, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                    _f4 = _mm512_add_ps(_f4, _c0_avx512);
                    _f5 = _mm512_add_ps(_f5, _c0_avx512);
                    _f6 = _mm512_add_ps(_f6, _c0_avx512);
                    _f7 = _mm512_add_ps(_f7, _c0_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1_avx512;
                    __m512 _c2_avx512;
                    __m512 _c3_avx512;
                    __m512 _c4_avx512;
                    __m512 _c5_avx512;
                    __m512 _c6_avx512;
                    __m512 _c7_avx512;
                    if (c_elempack == 8)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        _c4_avx512 = _mm512_loadu_ps(pC + 64);
                        _c5_avx512 = _mm512_loadu_ps(pC + 80);
                        _c6_avx512 = _mm512_loadu_ps(pC + 96);
                        _c7_avx512 = _mm512_loadu_ps(pC + 112);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c4_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c0_avx512, _c4_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c1_avx512, _c5_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c1_avx512, _c5_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c2_avx512, _c6_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c2_avx512, _c6_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c3_avx512, _c7_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c3_avx512, _c7_avx512, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0_avx512 = _tmp0;
                        _c1_avx512 = _tmp1;
                        _c2_avx512 = _tmp2;
                        _c3_avx512 = _tmp3;
                        _c4_avx512 = _tmp4;
                        _c5_avx512 = _tmp5;
                        _c6_avx512 = _tmp6;
                        _c7_avx512 = _tmp7;

                        pC += 128;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        _c4_avx512 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c6_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 32);
                        _c7_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 48);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c2_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c0_avx512, _c2_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c1_avx512, _c3_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c1_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c4_avx512, _c6_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c4_avx512, _c6_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c5_avx512, _c7_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c5_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 3, 1));

                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(3, 1, 3, 1));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(3, 1, 3, 1));
                        _c7_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                        _c0_avx512 = _mm512_shuffle_f32x4(_c0_avx512, _c0_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_c1_avx512, _c1_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_c2_avx512, _c2_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_c3_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c4_avx512 = _mm512_shuffle_f32x4(_c4_avx512, _c4_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_c5_avx512, _c5_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_c6_avx512, _c6_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c7_avx512 = _mm512_shuffle_f32x4(_c7_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 2, 0));

                        pC += 64;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                        _c2_avx512 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3_avx512 = _mm512_loadu_ps(pC + c_hstep * 3);
                        _c4_avx512 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5_avx512 = _mm512_loadu_ps(pC + c_hstep * 5);
                        _c6_avx512 = _mm512_loadu_ps(pC + c_hstep * 6);
                        _c7_avx512 = _mm512_loadu_ps(pC + c_hstep * 7);

                        __m512 _tmp0 = _mm512_unpacklo_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp1 = _mm512_unpacklo_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp2 = _mm512_unpacklo_ps(_c4_avx512, _c5_avx512);
                        __m512 _tmp3 = _mm512_unpacklo_ps(_c6_avx512, _c7_avx512);
                        __m512 _tmp4 = _mm512_unpackhi_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp5 = _mm512_unpackhi_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp6 = _mm512_unpackhi_ps(_c4_avx512, _c5_avx512);
                        __m512 _tmp7 = _mm512_unpackhi_ps(_c6_avx512, _c7_avx512);

                        _c0_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c1_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c2_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c3_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c4_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                        _c5_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));
                        _c6_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                        _c7_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));

                        _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp1 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp2 = _mm512_shuffle_f32x4(_c4_avx512, _c5_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp3 = _mm512_shuffle_f32x4(_c6_avx512, _c7_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp4 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp5 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp6 = _mm512_shuffle_f32x4(_c4_avx512, _c5_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp7 = _mm512_shuffle_f32x4(_c6_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 3, 1));

                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                        _c4_avx512 = _mm512_shuffle_f32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                        _c7_avx512 = _mm512_shuffle_f32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                        pC += 16;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                        _f2 = _mm512_add_ps(_f2, _c2_avx512);
                        _f3 = _mm512_add_ps(_f3, _c3_avx512);
                        _f4 = _mm512_add_ps(_f4, _c4_avx512);
                        _f5 = _mm512_add_ps(_f5, _c5_avx512);
                        _f6 = _mm512_add_ps(_f6, _c6_avx512);
                        _f7 = _mm512_add_ps(_f7, _c7_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2_avx512, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3_avx512, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4_avx512, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5_avx512, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6_avx512, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7_avx512, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _cc = _mm512_loadu_ps(pC);
                    _cc = _mm512_mul_ps(_cc, _mm512_set1_ps(beta));
                    __m512 _cc0 = _mm512_permute_ps(_cc, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _cc1 = _mm512_permute_ps(_cc, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _cc2 = _mm512_permute_ps(_cc, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _cc3 = _mm512_permute_ps(_cc, _MM_SHUFFLE(3, 3, 3, 3));

                    _c0_avx512 = _mm512_shuffle_f32x4(_cc0, _cc0, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c1_avx512 = _mm512_shuffle_f32x4(_cc1, _cc1, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c2_avx512 = _mm512_shuffle_f32x4(_cc2, _cc2, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c3_avx512 = _mm512_shuffle_f32x4(_cc3, _cc3, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c4_avx512 = _mm512_shuffle_f32x4(_cc0, _cc0, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c5_avx512 = _mm512_shuffle_f32x4(_cc1, _cc1, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c6_avx512 = _mm512_shuffle_f32x4(_cc2, _cc2, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c7_avx512 = _mm512_shuffle_f32x4(_cc3, _cc3, _MM_SHUFFLE(3, 3, 1, 1));

                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    _f2 = _mm512_add_ps(_f2, _c2_avx512);
                    _f3 = _mm512_add_ps(_f3, _c3_avx512);
                    _f4 = _mm512_add_ps(_f4, _c4_avx512);
                    _f5 = _mm512_add_ps(_f5, _c5_avx512);
                    _f6 = _mm512_add_ps(_f6, _c6_avx512);
                    _f7 = _mm512_add_ps(_f7, _c7_avx512);

                    pC += 16;
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

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);

                    _mm256_store_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_store_ps(p0 + 8, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_store_ps(p0 + 16, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_store_ps(p0 + 16 + 8, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_store_ps(p0 + 16 * 2, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_store_ps(p0 + 16 * 2 + 8, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_store_ps(p0 + 16 * 3, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_store_ps(p0 + 16 * 3 + 8, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_store_ps(p0 + 16 * 4, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_store_ps(p0 + 16 * 4 + 8, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_store_ps(p0 + 16 * 5, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_store_ps(p0 + 16 * 5 + 8, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_store_ps(p0 + 16 * 6, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_store_ps(p0 + 16 * 6 + 8, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_store_ps(p0 + 16 * 7, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_store_ps(p0 + 16 * 7 + 8, _mm512_extractf32x8_ps(_f7, 1));
                }
                if (out_elempack == 8)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 16 * 2, _f2);
                    _mm512_storeu_ps(p0 + 16 * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 2, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 3, _f7);
                }
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    transpose16x4_ps(_f4, _f5, _f6, _f7);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 16, _f7);
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + out_hstep, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 2, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 6, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 10, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 14, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x8_ps(_f7, 1));
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(p0, _tmp0);
                    _mm512_storeu_ps(p0 + 16, _tmp1);
                    _mm512_storeu_ps(p0 + 32, _tmp2);
                    _mm512_storeu_ps(p0 + 48, _tmp3);
                    _mm512_storeu_ps(p0 + 64, _tmp4);
                    _mm512_storeu_ps(p0 + 80, _tmp5);
                    _mm512_storeu_ps(p0 + 96, _tmp6);
                    _mm512_storeu_ps(p0 + 112, _tmp7);
                    p0 += 128;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _f4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f5 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _f7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 32, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 48, _f7);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f1);
                    __m512 _tmp1 = _mm512_unpacklo_ps(_f2, _f3);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_f4, _f5);
                    __m512 _tmp3 = _mm512_unpacklo_ps(_f6, _f7);
                    __m512 _tmp4 = _mm512_unpackhi_ps(_f0, _f1);
                    __m512 _tmp5 = _mm512_unpackhi_ps(_f2, _f3);
                    __m512 _tmp6 = _mm512_unpackhi_ps(_f4, _f5);
                    __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f7);

                    _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f1 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _f2 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                    _f5 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));
                    _f6 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                    _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));

                    _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _f4 = _mm512_shuffle_f32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _f5 = _mm512_shuffle_f32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _f6 = _mm512_shuffle_f32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _f7 = _mm512_shuffle_f32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 7, _f7);

                    p0 += 16;
                }
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 8)));
            __m256 _f2 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 16)));
            __m256 _f3 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 24)));
            __m256 _f4 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 32)));
            __m256 _f5 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 40)));
            __m256 _f6 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 48)));
            __m256 _f7 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 56)));
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
#else  // __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            __m256 _f2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 16)));
            __m256 _f3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 24)));
            __m256 _f4 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp1));
            __m256 _f5 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp1 + 8)));
            __m256 _f6 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp1 + 16)));
            __m256 _f7 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp1 + 24)));
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
                __m256 _tmp0 = _f0;
                __m256 _tmp1 = _f1;
                __m256 _tmp2 = _mm256_shuffle_ps(_f2, _f2, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _f4;
                __m256 _tmp5 = _f5;
                __m256 _tmp6 = _mm256_shuffle_ps(_f6, _f6, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));

                _f0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 2, 0, 0));
                _f1 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                _f2 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                _f3 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                _f4 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                _f5 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                _f6 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 1));
                _f7 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                _tmp0 = _mm256_unpacklo_ps(_f0, _f3);
                _tmp1 = _mm256_unpacklo_ps(_f1, _f2);
                _tmp2 = _mm256_unpackhi_ps(_f1, _f2);
                _tmp3 = _mm256_unpackhi_ps(_f0, _f3);
                _tmp4 = _mm256_unpacklo_ps(_f4, _f7);
                _tmp5 = _mm256_unpacklo_ps(_f5, _f6);
                _tmp6 = _mm256_unpackhi_ps(_f5, _f6);
                _tmp7 = _mm256_unpackhi_ps(_f4, _f7);

                _f0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));
                _f7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));

                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
            }
#endif // __AVX2__

            _f0 = _mm256_mul_ps(_f0, _descale);
            _f1 = _mm256_mul_ps(_f1, _descale);
            _f2 = _mm256_mul_ps(_f2, _descale);
            _f3 = _mm256_mul_ps(_f3, _descale);
            _f4 = _mm256_mul_ps(_f4, _descale);
            _f5 = _mm256_mul_ps(_f5, _descale);
            _f6 = _mm256_mul_ps(_f6, _descale);
            _f7 = _mm256_mul_ps(_f7, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c0);
                    _f6 = _mm256_add_ps(_f6, _c0);
                    _f7 = _mm256_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c0);
                    _f6 = _mm256_add_ps(_f6, _c0);
                    _f7 = _mm256_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    __m256 _c4;
                    __m256 _c5;
                    __m256 _c6;
                    __m256 _c7;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        _c2 = _mm256_loadu_ps(pC + 16);
                        _c3 = _mm256_loadu_ps(pC + 24);
                        _c4 = _mm256_loadu_ps(pC + 32);
                        _c5 = _mm256_loadu_ps(pC + 40);
                        _c6 = _mm256_loadu_ps(pC + 48);
                        _c7 = _mm256_loadu_ps(pC + 56);
                        pC += 64;
                    }
                    else if (c_elempack == 4)
                    {
                        __m256 _tmp0 = _mm256_loadu_ps(pC);
                        __m256 _tmp1 = _mm256_loadu_ps(pC + 8);
                        __m256 _tmp2 = _mm256_loadu_ps(pC + 16);
                        __m256 _tmp3 = _mm256_loadu_ps(pC + 24);
                        __m256 _tmp4 = _mm256_loadu_ps(pC + c_hstep * 4);
                        __m256 _tmp5 = _mm256_loadu_ps(pC + c_hstep * 4 + 8);
                        __m256 _tmp6 = _mm256_loadu_ps(pC + c_hstep * 4 + 16);
                        __m256 _tmp7 = _mm256_loadu_ps(pC + c_hstep * 4 + 24);
                        _c0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                        _c2 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                        _c3 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                        _c4 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                        _c5 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 1));
                        _c6 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                        _c7 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + c_hstep);
                        _c2 = _mm256_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm256_loadu_ps(pC + c_hstep * 3);
                        _c4 = _mm256_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm256_loadu_ps(pC + c_hstep * 5);
                        _c6 = _mm256_loadu_ps(pC + c_hstep * 6);
                        _c7 = _mm256_loadu_ps(pC + c_hstep * 7);
                        transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        pC += 8;
                    }
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
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_comp_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm256_comp_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm256_comp_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm256_comp_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm256_comp_fmadd_ps(_c7, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    __m256 _c2 = _mm256_set1_ps(pC[2] * beta);
                    __m256 _c3 = _mm256_set1_ps(pC[3] * beta);

                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    _f2 = _mm256_add_ps(_f2, _c2);
                    _f3 = _mm256_add_ps(_f3, _c3);

                    _c0 = _mm256_set1_ps(pC[4] * beta);
                    _c1 = _mm256_set1_ps(pC[5] * beta);
                    _c2 = _mm256_set1_ps(pC[6] * beta);
                    _c3 = _mm256_set1_ps(pC[7] * beta);

                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c1);
                    _f6 = _mm256_add_ps(_f6, _c2);
                    _f7 = _mm256_add_ps(_f7, _c3);
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

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm256_store_ps(p0, _f0);
                    _mm256_store_ps(p0 + 8, _f1);
                    _mm256_store_ps(p0 + 16, _f2);
                    _mm256_store_ps(p0 + 24, _f3);
                    _mm256_store_ps(p0 + 32, _f4);
                    _mm256_store_ps(p0 + 40, _f5);
                    _mm256_store_ps(p0 + 48, _f6);
                    _mm256_store_ps(p0 + 56, _f7);
                }
                if (out_elempack == 4)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    transpose8x4_ps(_f4, _f5, _f6, _f7);
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 8, _f5);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 16, _f6);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 24, _f7);
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + out_hstep, _f1);
                    _mm256_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm256_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm256_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm256_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm256_storeu_ps(p0 + out_hstep * 7, _f7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(p0, _f0);
                    _mm256_store_ps(p0 + 8, _f1);
                    _mm256_store_ps(p0 + 16, _f2);
                    _mm256_store_ps(p0 + 24, _f3);
                    _mm256_store_ps(p0 + 32, _f4);
                    _mm256_store_ps(p0 + 40, _f5);
                    _mm256_store_ps(p0 + 48, _f6);
                    _mm256_store_ps(p0 + 56, _f7);
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_f2, _f3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_f4, _f5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_f6, _f7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp4 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp5 = _mm256_permute2f128_ps(_f2, _f3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp6 = _mm256_permute2f128_ps(_f4, _f5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp7 = _mm256_permute2f128_ps(_f6, _f7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(p0, _tmp0);
                    _mm256_storeu_ps(p0 + 8, _tmp1);
                    _mm256_storeu_ps(p0 + 16, _tmp2);
                    _mm256_storeu_ps(p0 + 24, _tmp3);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _tmp4);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 8, _tmp5);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 16, _tmp6);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 24, _tmp7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
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
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 8)));
            __m256 _f2 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 16)));
            __m256 _f3 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 24)));
            pp += 32;
#else
            __m256 _f01l = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f23l = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            __m256 _f01h = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp1));
            __m256 _f23h = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp1 + 8)));
            __m256 _f0 = _mm256_permute2f128_ps(_f01l, _f01h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _f1 = _mm256_permute2f128_ps(_f01l, _f01h, _MM_SHUFFLE(0, 3, 0, 1));
            __m256 _f2 = _mm256_permute2f128_ps(_f23l, _f23h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _f3 = _mm256_permute2f128_ps(_f23l, _f23h, _MM_SHUFFLE(0, 3, 0, 1));
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
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp0 = _mm256_unpacklo_ps(_f0, _f3);
                __m256 _tmp1 = _mm256_unpackhi_ps(_f0, _f3);
                __m256 _tmp2 = _mm256_unpacklo_ps(_f2, _f1);
                __m256 _tmp3 = _mm256_unpackhi_ps(_f2, _f1);
                _f0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _f1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _f2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _f3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm256_mul_ps(_f0, _descale);
            _f1 = _mm256_mul_ps(_f1, _descale);
            _f2 = _mm256_mul_ps(_f2, _descale);
            _f3 = _mm256_mul_ps(_f3, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        _c2 = _mm256_loadu_ps(pC + 16);
                        _c3 = _mm256_loadu_ps(pC + 24);
                        pC += 32;
                    }
                    else if (c_elempack == 4)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + 8);
                        __m256 _cc2 = _mm256_loadu_ps(pC + c_hstep * 4);
                        __m256 _cc3 = _mm256_loadu_ps(pC + c_hstep * 4 + 8);
                        _c0 = _mm256_permute2f128_ps(_cc0, _cc2, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_cc0, _cc2, _MM_SHUFFLE(0, 3, 0, 1));
                        _c2 = _mm256_permute2f128_ps(_cc1, _cc3, _MM_SHUFFLE(0, 2, 0, 0));
                        _c3 = _mm256_permute2f128_ps(_cc1, _cc3, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        // __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        // _c0 = _mm256_i32gather_ps(pC, _vindex, c_hstep * sizeof(float));
                        // _c1 = _mm256_i32gather_ps(pC + 1, _vindex, c_hstep * sizeof(float));
                        // _c2 = _mm256_i32gather_ps(pC + 2, _vindex, c_hstep * sizeof(float));
                        // _c3 = _mm256_i32gather_ps(pC + 3, _vindex, c_hstep * sizeof(float));

                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 2);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 3);
                        __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 5);
                        __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 6);
                        __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 7);
                        _MM_TRANSPOSE4_PS(_cc0, _cc1, _cc2, _cc3);
                        _MM_TRANSPOSE4_PS(_cc4, _cc5, _cc6, _cc7);

                        _c0 = combine4x2_ps(_cc0, _cc4);
                        _c1 = combine4x2_ps(_cc1, _cc5);
                        _c2 = combine4x2_ps(_cc2, _cc6);
                        _c3 = combine4x2_ps(_cc3, _cc7);

                        pC += 4;
                    }
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
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    __m256 _c2 = _mm256_set1_ps(pC[2] * beta);
                    __m256 _c3 = _mm256_set1_ps(pC[3] * beta);

                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    _f2 = _mm256_add_ps(_f2, _c2);
                    _f3 = _mm256_add_ps(_f3, _c3);
                    pC += 4;
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

            if (output_transpose)
            {
#if !(defined(__x86_64__) || defined(_M_X64))
#if __AVX__
#if __AVX512F__
                if (out_elempack == 16)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    const int jj_m16 = jj % 16;
                    float* p1 = p0 - out_hstep * jj_m16 + jj_m16;
                    _mm_store_ps(p1, _mm256_extractf128_ps(_f0, 0));
                    _mm_store_ps(p1 + 16, _mm256_extractf128_ps(_f0, 1));
                    _mm_store_ps(p1 + 32, _mm256_extractf128_ps(_f1, 0));
                    _mm_store_ps(p1 + 48, _mm256_extractf128_ps(_f1, 1));
                    _mm_store_ps(p1 + 64, _mm256_extractf128_ps(_f2, 0));
                    _mm_store_ps(p1 + 80, _mm256_extractf128_ps(_f2, 1));
                    _mm_store_ps(p1 + 96, _mm256_extractf128_ps(_f3, 0));
                    _mm_store_ps(p1 + 112, _mm256_extractf128_ps(_f3, 1));
                }
#endif // __AVX512F__
                if (out_elempack == 8)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    const int jj_m8 = jj % 8;
                    float* p1 = p0 - out_hstep * jj_m8 + jj_m8;
                    _mm_store_ps(p1, _mm256_extractf128_ps(_f0, 0));
                    _mm_store_ps(p1 + 8, _mm256_extractf128_ps(_f0, 1));
                    _mm_store_ps(p1 + 16, _mm256_extractf128_ps(_f1, 0));
                    _mm_store_ps(p1 + 24, _mm256_extractf128_ps(_f1, 1));
                    _mm_store_ps(p1 + 32, _mm256_extractf128_ps(_f2, 0));
                    _mm_store_ps(p1 + 40, _mm256_extractf128_ps(_f2, 1));
                    _mm_store_ps(p1 + 48, _mm256_extractf128_ps(_f3, 0));
                    _mm_store_ps(p1 + 56, _mm256_extractf128_ps(_f3, 1));
                }
#endif // __AVX__
#endif // !(defined(__x86_64__) || defined(_M_X64))
                if (out_elempack == 4)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + out_hstep, _f1);
                    _mm256_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm256_storeu_ps(p0 + out_hstep * 3, _f3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(p0, _f0);
                    _mm256_store_ps(p0 + 8, _f1);
                    _mm256_store_ps(p0 + 16, _f2);
                    _mm256_store_ps(p0 + 24, _f3);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_f2, _f3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_f2, _f3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(p0, _tmp0);
                    _mm256_storeu_ps(p0 + 8, _tmp1);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _tmp2);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 8, _tmp3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    _mm_storeu_ps(p0, _mm256_extractf128_ps(_f0, 0));
                    _mm_storeu_ps(p0 + out_hstep, _mm256_extractf128_ps(_f0, 1));
                    _mm_storeu_ps(p0 + out_hstep * 2, _mm256_extractf128_ps(_f1, 0));
                    _mm_storeu_ps(p0 + out_hstep * 3, _mm256_extractf128_ps(_f1, 1));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm256_extractf128_ps(_f2, 0));
                    _mm_storeu_ps(p0 + out_hstep * 5, _mm256_extractf128_ps(_f2, 1));
                    _mm_storeu_ps(p0 + out_hstep * 6, _mm256_extractf128_ps(_f3, 0));
                    _mm_storeu_ps(p0 + out_hstep * 7, _mm256_extractf128_ps(_f3, 1));
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(pp + 8)));
            pp += 16;
#else
            __m256 _f01l = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f01h = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp1));
            __m256 _f0 = _mm256_permute2f128_ps(_f01l, _f01h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _f1 = _mm256_permute2f128_ps(_f01l, _f01h, _MM_SHUFFLE(0, 3, 0, 1));
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
                __m256 _tmp0 = _mm256_shuffle_ps(_f0, _f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m256 _tmp1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm256_unpacklo_ps(_tmp0, _tmp1);
                _f1 = _mm256_unpackhi_ps(_tmp0, _tmp1);
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm256_mul_ps(_f0, _descale);
            _f1 = _mm256_mul_ps(_f1, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        pC += 16;
                    }
                    else if (c_elempack == 4)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep * 4);
                        _c0 = _mm256_permute2f128_ps(_cc0, _cc1, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_cc0, _cc1, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(c_hstep));
                        _c0 = _mm256_i32gather_ps(pC, _vindex, sizeof(float));
                        _c1 = _mm256_i32gather_ps(pC + 1, _vindex, sizeof(float));
#else
                        _c0 = _mm256_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3], pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]);
                        _c1 = _mm256_setr_ps(pC[1], pC[c_hstep + 1], pC[c_hstep * 2 + 1], pC[c_hstep * 3 + 1], pC[c_hstep * 4 + 1], pC[c_hstep * 5 + 1], pC[c_hstep * 6 + 1], pC[c_hstep * 7 + 1]);
#endif
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                _mm256_storeu_ps(p0, _f0);
                _mm256_storeu_ps(p0 + out_hstep, _f1);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(p0, _tmp0);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _tmp1);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(out_hstep));
                    _mm256_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    _mm256_i32scatter_ps(p0 + 1, _vindex, _f1, sizeof(float));
#else
                    float sum0[8];
                    float sum1[8];
                    _mm256_storeu_ps(sum0, _f0);
                    _mm256_storeu_ps(sum1, _f1);

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
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)pp));
            pp += 8;
#else
            __m128i _f0l = _mm_load_si128((const __m128i*)pp);
            __m128i _f0h = _mm_load_si128((const __m128i*)pp1);
            __m256 _f0 = _mm256_cvtepi32_ps(combine4x2_epi32(_f0l, _f0h));
            pp += 4;
            pp1 += 4;
#endif

            _f0 = _mm256_mul_ps(_f0, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        pC += 8;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4);
                        _c0 = combine4x2_ps(_cc0, _cc1);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(c_hstep));
                        _c0 = _mm256_i32gather_ps(pC, _vindex, sizeof(float));
#else
                        _c0 = _mm256_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3], pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]);
#endif
                        pC += 1;
                    }
                    _f0 = _mm256_comp_fmadd_ps(_c0, _mm256_set1_ps(beta), _f0);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    _f0 = _mm256_add_ps(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = _mm256_mul_ps(_f0, _mm256_set1_ps(alpha));

            if (output_transpose)
            {
                _mm256_storeu_ps(p0, _f0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _f0);
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _mm256_extractf128_ps(_f0, 0));
                    _mm_store_ps(p0 + out_hstep * 4, _mm256_extractf128_ps(_f0, 1));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(out_hstep));
                    _mm256_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
#else
                    float sum0[8];
                    _mm256_storeu_ps(sum0, _f0);
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
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m128 _descale = _mm_load_ps((const float*)descales + i + ii);
#if __AVX512F__
        __m512 _descale_avx512 = _mm512_broadcast_f32x4(_descale);
#endif

        __m128 _c0 = _mm_set1_ps(0.f);
#if __AVX512F__
        __m512 _c0_avx512 = _mm512_set1_ps(0.f);
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm_set1_ps(pC[0] * beta);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm_loadu_ps(pC);
                _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
#if __AVX512F__
                _c0_avx512 = _mm512_broadcast_f32x4(_c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 48)));

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
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp2 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm512_mul_ps(_f0, _descale_avx512);
            _f1 = _mm512_mul_ps(_f1, _descale_avx512);
            _f2 = _mm512_mul_ps(_f2, _descale_avx512);
            _f3 = _mm512_mul_ps(_f3, _descale_avx512);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1_avx512;
                    __m512 _c2_avx512;
                    __m512 _c3_avx512;
                    if (c_elempack == 4)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        pC += 64;

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                        _c2_avx512 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3_avx512 = _mm512_loadu_ps(pC + c_hstep * 3);
                        pC += 16;

                        __m512 _tmp0 = _mm512_unpacklo_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp1 = _mm512_unpacklo_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp2 = _mm512_unpackhi_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp3 = _mm512_unpackhi_ps(_c2_avx512, _c3_avx512);
                        _c0_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c1_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c2_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c3_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                        _f2 = _mm512_add_ps(_f2, _c2_avx512);
                        _f3 = _mm512_add_ps(_f3, _c3_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2_avx512, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3_avx512, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _cc = _mm512_loadu_ps(pC);
                    _cc = _mm512_mul_ps(_cc, _mm512_set1_ps(beta));
                    _c0_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _c1_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _c2_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _c3_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(3, 3, 3, 3));

                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    _f2 = _mm512_add_ps(_f2, _c2_avx512);
                    _f3 = _mm512_add_ps(_f3, _c3_avx512);

                    pC += 16;
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

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    _mm_store_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_store_ps(p0 + 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_store_ps(p0 + 8, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_store_ps(p0 + 12, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_store_ps(p0 + 16, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_store_ps(p0 + 16 + 4, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_store_ps(p0 + 16 + 8, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_store_ps(p0 + 16 + 12, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_store_ps(p0 + 32, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_store_ps(p0 + 32 + 4, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_store_ps(p0 + 32 + 8, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_store_ps(p0 + 32 + 12, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_store_ps(p0 + 48, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_store_ps(p0 + 48 + 4, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_store_ps(p0 + 48 + 8, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_store_ps(p0 + 48 + 12, _mm512_extractf32x4_ps(_f3, 3));
                }
                if (out_elempack == 8)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    _mm_store_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_store_ps(p0 + 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_store_ps(p0 + 8, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_store_ps(p0 + 12, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_store_ps(p0 + 16, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_store_ps(p0 + 16 + 4, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_store_ps(p0 + 16 + 8, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_store_ps(p0 + 16 + 12, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_store_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_store_ps(p0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_store_ps(p0 + out_hstep * 8 + 8, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_store_ps(p0 + out_hstep * 8 + 12, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_store_ps(p0 + out_hstep * 8 + 16, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_store_ps(p0 + out_hstep * 8 + 16 + 4, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_store_ps(p0 + out_hstep * 8 + 16 + 8, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_store_ps(p0 + out_hstep * 8 + 16 + 12, _mm512_extractf32x4_ps(_f3, 3));
                }
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _f3);
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p0 + out_hstep, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p0 + out_hstep * 2, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p0 + out_hstep * 6, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p0 + out_hstep * 10, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_storeu_ps(p0 + out_hstep * 14, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x4_ps(_f3, 3));
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f1);
                    __m512 _tmp1 = _mm512_unpacklo_ps(_f2, _f3);
                    __m512 _tmp2 = _mm512_unpackhi_ps(_f0, _f1);
                    __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f3);
                    _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    p0 += 16;
                }
            }

            pp += 64;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)pp));
            __m128 _f1 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 4)));
            __m128 _f2 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 8)));
            __m128 _f3 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 12)));
            __m128 _f4 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 16)));
            __m128 _f5 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 20)));
            __m128 _f6 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 24)));
            __m128 _f7 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 28)));

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
                _f4 = _mm_shuffle_ps(_f4, _f4, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f6 = _mm_shuffle_ps(_f6, _f6, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f6);
                __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f6);
                __m128 _tmp2 = _mm_unpacklo_ps(_f1, _f7);
                __m128 _tmp3 = _mm_unpackhi_ps(_f1, _f7);
                __m128 _tmp4 = _mm_unpacklo_ps(_f2, _f4);
                __m128 _tmp5 = _mm_unpackhi_ps(_f2, _f4);
                __m128 _tmp6 = _mm_unpacklo_ps(_f3, _f5);
                __m128 _tmp7 = _mm_unpackhi_ps(_f3, _f5);
                _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp4)));
                _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp4)));
                _f2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp5), _mm_castps_pd(_tmp1)));
                _f3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp5), _mm_castps_pd(_tmp1)));
                _f4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp2), _mm_castps_pd(_tmp6)));
                _f5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp2), _mm_castps_pd(_tmp6)));
                _f6 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp7), _mm_castps_pd(_tmp3)));
                _f7 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp7), _mm_castps_pd(_tmp3)));
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm_mul_ps(_f0, _descale);
            _f1 = _mm_mul_ps(_f1, _descale);
            _f2 = _mm_mul_ps(_f2, _descale);
            _f3 = _mm_mul_ps(_f3, _descale);
            _f4 = _mm_mul_ps(_f4, _descale);
            _f5 = _mm_mul_ps(_f5, _descale);
            _f6 = _mm_mul_ps(_f6, _descale);
            _f7 = _mm_mul_ps(_f7, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c0);
                    _f6 = _mm_add_ps(_f6, _c0);
                    _f7 = _mm_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c0);
                    _f6 = _mm_add_ps(_f6, _c0);
                    _f7 = _mm_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        _c2 = _mm_loadu_ps(pC + 8);
                        _c3 = _mm_loadu_ps(pC + 12);
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + c_hstep);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                    }
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
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC + 16);
                        _c1 = _mm_loadu_ps(pC + 20);
                        _c2 = _mm_loadu_ps(pC + 24);
                        _c3 = _mm_loadu_ps(pC + 28);
                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC + 4);
                        _c1 = _mm_loadu_ps(pC + c_hstep + 4);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2 + 4);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3 + 4);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f4 = _mm_add_ps(_f4, _c0);
                        _f5 = _mm_add_ps(_f5, _c1);
                        _f6 = _mm_add_ps(_f6, _c2);
                        _f7 = _mm_add_ps(_f7, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f4 = _mm_comp_fmadd_ps(_c0, _beta, _f4);
                        _f5 = _mm_comp_fmadd_ps(_c1, _beta, _f5);
                        _f6 = _mm_comp_fmadd_ps(_c2, _beta, _f6);
                        _f7 = _mm_comp_fmadd_ps(_c3, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    __m128 _c2 = _mm_set1_ps(pC[2] * beta);
                    __m128 _c3 = _mm_set1_ps(pC[3] * beta);

                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c2);
                    _f3 = _mm_add_ps(_f3, _c3);

                    _c0 = _mm_set1_ps(pC[4] * beta);
                    _c1 = _mm_set1_ps(pC[5] * beta);
                    _c2 = _mm_set1_ps(pC[6] * beta);
                    _c3 = _mm_set1_ps(pC[7] * beta);

                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c1);
                    _f6 = _mm_add_ps(_f6, _c2);
                    _f7 = _mm_add_ps(_f7, _c3);
                    pC += 8;
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

            if (output_transpose)
            {
#if __AVX__
                if (out_elempack == 8)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _MM_TRANSPOSE4_PS(_f4, _f5, _f6, _f7);
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f4);
                    _mm_store_ps(p0 + 8, _f1);
                    _mm_store_ps(p0 + 12, _f5);
                    _mm_store_ps(p0 + 16, _f2);
                    _mm_store_ps(p0 + 20, _f6);
                    _mm_store_ps(p0 + 24, _f3);
                    _mm_store_ps(p0 + 28, _f7);
                }
#endif // __AVX__
                if (out_elempack == 4)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _MM_TRANSPOSE4_PS(_f4, _f5, _f6, _f7);
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f1);
                    _mm_store_ps(p0 + 8, _f2);
                    _mm_store_ps(p0 + 12, _f3);
                    _mm_store_ps(p0 + out_hstep * 4, _f4);
                    _mm_store_ps(p0 + out_hstep * 4 + 4, _f5);
                    _mm_store_ps(p0 + out_hstep * 4 + 8, _f6);
                    _mm_store_ps(p0 + out_hstep * 4 + 12, _f7);
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + out_hstep, _f1);
                    _mm_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm_storeu_ps(p0 + out_hstep * 7, _f7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f1);
                    _mm_store_ps(p0 + 8, _f2);
                    _mm_store_ps(p0 + 12, _f3);
                    _mm_store_ps(p0 + 16, _f4);
                    _mm_store_ps(p0 + 20, _f5);
                    _mm_store_ps(p0 + 24, _f6);
                    _mm_store_ps(p0 + 28, _f7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _MM_TRANSPOSE4_PS(_f4, _f5, _f6, _f7);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f4);
                    _mm_storeu_ps(p0 + out_hstep, _f1);
                    _mm_storeu_ps(p0 + out_hstep + 4, _f5);
                    _mm_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 2 + 4, _f6);
                    _mm_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm_storeu_ps(p0 + out_hstep * 3 + 4, _f7);
                    p0 += 8;
                }
            }

            pp += 32;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)pp));
            __m128 _f1 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 4)));
            __m128 _f2 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 8)));
            __m128 _f3 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 12)));

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
            }

            _f0 = _mm_mul_ps(_f0, _descale);
            _f1 = _mm_mul_ps(_f1, _descale);
            _f2 = _mm_mul_ps(_f2, _descale);
            _f3 = _mm_mul_ps(_f3, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        _c2 = _mm_loadu_ps(pC + 8);
                        _c3 = _mm_loadu_ps(pC + 12);
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + c_hstep);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                        pC += 4;
                    }
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
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    __m128 _c2 = _mm_set1_ps(pC[2] * beta);
                    __m128 _c3 = _mm_set1_ps(pC[3] * beta);

                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c2);
                    _f3 = _mm_add_ps(_f3, _c3);
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

            if (output_transpose)
            {
#if !(defined(__x86_64__) || defined(_M_X64))
#if __AVX__
#if __AVX512F__
                if (out_elempack == 16)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    const int jj_m16 = jj % 16;
                    float* p1 = p0 - out_hstep * jj_m16 + jj_m16;
                    _mm_store_ps(p1, _f0);
                    _mm_store_ps(p1 + 16, _f1);
                    _mm_store_ps(p1 + 32, _f2);
                    _mm_store_ps(p1 + 48, _f3);
                }
#endif // __AVX512F__
                if (out_elempack == 8)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    const int jj_m8 = jj % 8;
                    float* p1 = p0 - out_hstep * jj_m8 + jj_m8;
                    _mm_store_ps(p1, _f0);
                    _mm_store_ps(p1 + 8, _f1);
                    _mm_store_ps(p1 + 16, _f2);
                    _mm_store_ps(p1 + 24, _f3);
                }
#endif // __AVX__
#endif // !(defined(__x86_64__) || defined(_M_X64))
                if (out_elempack == 4)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f1);
                    _mm_store_ps(p0 + 8, _f2);
                    _mm_store_ps(p0 + 12, _f3);
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + out_hstep, _f1);
                    _mm_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 3, _f3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f1);
                    _mm_store_ps(p0 + 8, _f2);
                    _mm_store_ps(p0 + 12, _f3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + out_hstep, _f1);
                    _mm_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 3, _f3);
                    p0 += 4;
                }
            }

            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)pp));
            __m128 _f1 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 4)));

            // from
            //      00 11 20 31
            //      01 10 21 30
            // to
            //      00 10 20 30
            //      01 11 21 31
            {
                __m128 _tmp0 = _mm_shuffle_ps(_f0, _f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m128 _tmp1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm_unpacklo_ps(_tmp0, _tmp1);
                _f1 = _mm_unpackhi_ps(_tmp0, _tmp1);
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm_mul_ps(_f0, _descale);
            _f1 = _mm_mul_ps(_f1, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                        _c1 = _mm_setr_ps(pC[1], pC[c_hstep + 1], pC[c_hstep * 2 + 1], pC[c_hstep * 3 + 1]);
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                _mm_storeu_ps(p0, _f0);
                _mm_storeu_ps(p0 + out_hstep, _f1);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f1);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m128i _vindex = _mm_mullo_epi32(_mm_setr_epi32(0, 1, 2, 3), _mm_set1_epi32(out_hstep));
                    _mm_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    _mm_i32scatter_ps(p0 + 1, _vindex, _f1, sizeof(float));
#else
                    float sum0[4];
                    float sum1[4];
                    _mm_storeu_ps(sum0, _f0);
                    _mm_storeu_ps(sum1, _f1);

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
            __m128 _f0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)pp));

            _f0 = _mm_mul_ps(_f0, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                        pC += 1;
                    }
                    _f0 = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f0);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    _f0 = _mm_add_ps(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));

            if (output_transpose)
            {
                _mm_storeu_ps(p0, _f0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _f0);
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m128i _vindex = _mm_mullo_epi32(_mm_setr_epi32(0, 1, 2, 3), _mm_set1_epi32(out_hstep));
                    _mm_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
#else
                    float sum0[4];
                    _mm_storeu_ps(sum0, _f0);
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
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            // out_elempack == 1
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float descale0 = descales[i + ii];
        const float descale1 = descales[i + ii + 1];
#if __SSE2__
        __m128 _descale0 = _mm_set1_ps(descale0);
        __m128 _descale1 = _mm_set1_ps(descale1);
#if __AVX512F__
        __m512 _descale0_avx512 = _mm512_set1_ps(descale0);
        __m512 _descale1_avx512 = _mm512_set1_ps(descale1);
#endif // __AVX512F__
#endif

        float c0 = 0.f;
        float c1 = 0.f;
#if __SSE2__
        __m128 _c0 = _mm_set1_ps(0.f);
        __m128 _c1 = _mm_set1_ps(0.f);
#if __AVX512F__
        __m512 _c0_avx512 = _mm512_set1_ps(0.f);
        __m512 _c1_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
                _c1 = _mm_set1_ps(c1);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
                _c1_avx512 = _mm512_set1_ps(c1);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 16)));

            // 00 11 02 13  04 15 06 17  08 19 0a 1b  0c 1d 0e 1f
            // 01 12 03 10  05 16 07 14  09 1a 0b 18  0d 1e 0f 1c

            __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f1);
            __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f1);

            _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
            _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));

            _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));

            _f0 = _mm512_mul_ps(_f0, _descale0_avx512);
            _f1 = _mm512_mul_ps(_f1, _descale1_avx512);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                    }
                    pC += 16;
                }
                if (broadcast_type_C == 4)
                {
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _c0_avx512 = _mm512_mul_ps(_c0_avx512, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(p0, _f0);
                    _mm512_store_ps(p0 + 16, _f1);
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_store_ps(p0 + 8, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_store_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_store_ps(p0 + out_hstep * 8 + 8, _mm512_extractf32x8_ps(_f1, 1));
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_store_ps(p0 + 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_store_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_store_ps(p0 + out_hstep * 4 + 4, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_store_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_store_ps(p0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_store_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_store_ps(p0 + out_hstep * 12 + 4, _mm512_extractf32x4_ps(_f1, 3));
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    _mm512_i32scatter_ps(p0 + 1, _vindex, _f1, sizeof(float));
                }
                p0 += out_hstep * 16;
            }
            else
            {
                _mm512_storeu_ps(p0, _f0);
                _mm512_storeu_ps(p0 + out_hstep, _f1);
                p0 += 16;
            }

            pp += 32;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)pp));
            __m128 _f1 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 4)));
            __m128 _f2 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 8)));
            __m128 _f3 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 12)));

            // 00 11 02 13
            // 04 15 06 17
            // 10 01 12 03
            // 14 05 16 07
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

            _f0 = _mm_mul_ps(_f0, _descale0);
            _f1 = _mm_mul_ps(_f1, _descale0);
            _f2 = _mm_mul_ps(_f2, _descale1);
            _f3 = _mm_mul_ps(_f3, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c1);
                    _f3 = _mm_add_ps(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + 4);
                    __m128 _c2 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _c3 = _mm_loadu_ps(pC + c_hstep + 4);
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
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + 4);
                    _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
                    _c1 = _mm_mul_ps(_c1, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c1);
                    pC += 8;
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

            if (output_transpose)
            {
#if __AVX__
                if (out_elempack == 8)
                {
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f1);
                    _mm_store_ps(p0 + 8, _f2);
                    _mm_store_ps(p0 + 12, _f3);
                }
#endif // __AVX__
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f2);
                    _mm_store_ps(p0 + out_hstep * 4, _f1);
                    _mm_store_ps(p0 + out_hstep * 4 + 4, _f3);
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    float sum2[4];
                    float sum3[4];
                    _mm_storeu_ps(sum0, _f0);
                    _mm_storeu_ps(sum1, _f1);
                    _mm_storeu_ps(sum2, _f2);
                    _mm_storeu_ps(sum3, _f3);

                    p0[0] = sum0[0];
                    p0[1] = sum2[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum2[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum2[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum2[3];
                    p0[out_hstep * 4] = sum1[0];
                    p0[out_hstep * 4 + 1] = sum3[0];
                    p0[out_hstep * 5] = sum1[1];
                    p0[out_hstep * 5 + 1] = sum3[1];
                    p0[out_hstep * 6] = sum1[2];
                    p0[out_hstep * 6 + 1] = sum3[2];
                    p0[out_hstep * 7] = sum1[3];
                    p0[out_hstep * 7 + 1] = sum3[3];
                }
                p0 += out_hstep * 8;
            }
            else
            {
                _mm_storeu_ps(p0, _f0);
                _mm_storeu_ps(p0 + 4, _f1);
                _mm_storeu_ps(p0 + out_hstep, _f2);
                _mm_storeu_ps(p0 + out_hstep + 4, _f3);
                p0 += 8;
            }

            pp += 16;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)pp));
            __m128 _f1 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(pp + 4)));

            // 00 11 02 13
            // 01 12 03 10
            __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f1);
            __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f1);

            _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
            _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp1), _mm_castps_pd(_tmp0)));

            _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(0, 3, 2, 1));

            _f0 = _mm_mul_ps(_f0, _descale0);
            _f1 = _mm_mul_ps(_f1, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_loadu_ps(pC);
                    _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
#if !(defined(__x86_64__) || defined(_M_X64))
#if __AVX__
#if __AVX512F__
                if (out_elempack == 16)
                {
                    const int jj_m16 = jj % 16;
                    float* p1 = p0 - out_hstep * jj_m16 + jj_m16;
                    _mm_store_ps(p1, _f0);
                    _mm_store_ps(p1 + 16, _f1);
                }
#endif // __AVX512F__
                if (out_elempack == 8)
                {
                    const int jj_m8 = jj % 8;
                    float* p1 = p0 - out_hstep * jj_m8 + jj_m8;
                    _mm_store_ps(p1, _f0);
                    _mm_store_ps(p1 + 8, _f1);
                }
#endif // __AVX__
#endif // !(defined(__x86_64__) || defined(_M_X64))
                if (out_elempack == 4)
                {
                    _mm_store_ps(p0, _f0);
                    _mm_store_ps(p0 + 4, _f1);
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    _mm_storeu_ps(sum0, _f0);
                    _mm_storeu_ps(sum1, _f1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];
                }
                p0 += out_hstep * 4;
            }
            else
            {
                _mm_storeu_ps(p0, _f0);
                _mm_storeu_ps(p0 + out_hstep, _f1);
                p0 += 4;
            }

            pp += 8;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f00 = pp[0] * descale0;
            float f01 = pp[1] * descale0;
            float f10 = pp[2] * descale1;
            float f11 = pp[3] * descale1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c0;
                    f11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c1;
                    f11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    f00 += pC[0] * beta;
                    f01 += pC[1] * beta;
                    f10 += pC[c_hstep] * beta;
                    f11 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    f00 += pC[0] * beta;
                    f01 += pC[1] * beta;
                    f10 += pC[0] * beta;
                    f11 += pC[1] * beta;
                    pC += 2;
                }
            }

            f00 *= alpha;
            f01 *= alpha;
            f10 *= alpha;
            f11 *= alpha;

            if (output_transpose)
            {
                p0[0] = f00;
                p0[1] = f10;
                p0[out_hstep] = f01;
                p0[out_hstep + 1] = f11;
                p0 += out_hstep * 2;
            }
            else
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
            float f0 = pp[0] * descale0;
            float f1 = pp[1] * descale1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    f0 += pC[0] * beta;
                    f1 += pC[c_hstep] * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += out_hstep;
            }
            else
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
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            // out_elempack == 1
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float descale = descales[i + ii];
#if __SSE2__
        __m128 _descale = _mm_set1_ps(descale);
#if __AVX512F__
        __m512 _descale_avx512 = _mm512_set1_ps(descale);
#endif // __AVX512F__
#endif

        float c0 = 0.f;
#if __SSE2__
        __m128 _c0 = _mm_set1_ps(0.f);
#if __AVX512F__
        __m512 _c0_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)pp)), _descale_avx512);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _f0 = _mm512_fmadd_ps(_c0_avx512, _mm512_set1_ps(beta), _f0);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                _f0 = _mm512_mul_ps(_f0, _mm512_set1_ps(alpha));
            }

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm512_storeu_ps(p0, _f0);
                }
                else
                {
                    if (out_elempack == 16)
                    {
                        _mm512_storeu_ps(p0, _f0);
                    }
                    if (out_elempack == 8)
                    {
                        _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                        _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    }
                    if (out_elempack == 4)
                    {
                        _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                        _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                        _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                        _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    }
                    if (out_elempack == 1)
                    {
                        __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                        _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    }
                }
                p0 += out_hstep * 16;
            }
            else
            {
                _mm512_storeu_ps(p0, _f0);
                p0 += 16;
            }

            pp += 16;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)pp)), _descale);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(pp + 4))), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + 4);
                    _f0 = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f0);
                    _f1 = _mm_comp_fmadd_ps(_c1, _mm_set1_ps(beta), _f1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                }
                else
                {
#if __AVX__
                    if (out_elempack == 8)
                    {
                        _mm_storeu_ps(p0, _f0);
                        _mm_storeu_ps(p0 + 4, _f1);
                    }
#endif // __AVX__
                    if (out_elempack == 4)
                    {
                        _mm_storeu_ps(p0, _f0);
                        _mm_storeu_ps(p0 + out_hstep * 4, _f1);
                    }
                    if (out_elempack == 1)
                    {
                        float sum0[4];
                        float sum1[4];
                        _mm_storeu_ps(sum0, _f0);
                        _mm_storeu_ps(sum1, _f1);

                        p0[0] = sum0[0];
                        p0[out_hstep] = sum0[1];
                        p0[out_hstep * 2] = sum0[2];
                        p0[out_hstep * 3] = sum0[3];
                        p0[out_hstep * 4] = sum1[0];
                        p0[out_hstep * 5] = sum1[1];
                        p0[out_hstep * 6] = sum1[2];
                        p0[out_hstep * 7] = sum1[3];
                    }
                }
                p0 += out_hstep * 8;
            }
            else
            {
                _mm_storeu_ps(p0, _f0);
                _mm_storeu_ps(p0 + 4, _f1);
                p0 += 8;
            }

            pp += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)pp)), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _f0 = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f0);
                    pC += 4;
                }
            }

            _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm_storeu_ps(p0, _f0);
                }
                else
                {
#if !(defined(__x86_64__) || defined(_M_X64))
#if __AVX__
#if __AVX512F__
                    if (out_elempack == 16)
                    {
                        _mm_storeu_ps(p0 - (jj % 16) / 4 * out_hstep * 4 + (jj % 16) / 4 * 4, _f0);
                    }
#endif // __AVX512F__
                    if (out_elempack == 8)
                    {
                        _mm_storeu_ps(p0 - (jj % 8) / 4 * out_hstep * 4 + (jj % 8) / 4 * 4, _f0);
                    }
#endif // __AVX__
#endif // !(defined(__x86_64__) || defined(_M_X64))
                    if (out_elempack == 4)
                    {
                        _mm_storeu_ps(p0, _f0);
                    }
                    if (out_elempack == 1)
                    {
                        float sum0[4];
                        _mm_storeu_ps(sum0, _f0);

                        p0[0] = sum0[0];
                        p0[out_hstep] = sum0[1];
                        p0[out_hstep * 2] = sum0[2];
                        p0[out_hstep * 3] = sum0[3];
                    }
                }
                p0 += out_hstep * 4;
            }
            else
            {
                _mm_storeu_ps(p0, _f0);
                p0 += 4;
            }

            pp += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0 = pp[0] * descale;
            float f1 = pp[1] * descale;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    f0 += pC[0] * beta;
                    f1 += pC[1] * beta;
                    pC += 2;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                p0[0] = f0;
                p0[out_hstep] = f1;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += 2;
            }

            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0] * descale;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    f0 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            p0[0] = f0;

            if (output_transpose)
            {
                p0 += out_hstep;
            }
            else
            {
                p0++;
            }

            pp += 1;
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        gemm_transB_packed_tile_int8_avx512vnni(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        gemm_transB_packed_tile_int8_avxvnniint8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        gemm_transB_packed_tile_int8_avxvnni(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        gemm_transB_packed_tile_int8_avx2(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_xop())
    {
        gemm_transB_packed_tile_int8_xop(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    // actually we only depend the global k==0 condition
    (void)i;
    (void)j;

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;
            __m512i _sum4;
            __m512i _sum5;
            __m512i _sum6;
            __m512i _sum7;
            __m512i _sum8;
            __m512i _sum9;
            __m512i _suma;
            __m512i _sumb;
            __m512i _sumc;
            __m512i _sumd;
            __m512i _sume;
            __m512i _sumf;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
                _sum4 = _mm512_setzero_si512();
                _sum5 = _mm512_setzero_si512();
                _sum6 = _mm512_setzero_si512();
                _sum7 = _mm512_setzero_si512();
                _sum8 = _mm512_setzero_si512();
                _sum9 = _mm512_setzero_si512();
                _suma = _mm512_setzero_si512();
                _sumb = _mm512_setzero_si512();
                _sumc = _mm512_setzero_si512();
                _sumd = _mm512_setzero_si512();
                _sume = _mm512_setzero_si512();
                _sumf = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
                _sum4 = _mm512_load_si512((const __m512i*)(outptr + 64));
                _sum5 = _mm512_load_si512((const __m512i*)(outptr + 80));
                _sum6 = _mm512_load_si512((const __m512i*)(outptr + 96));
                _sum7 = _mm512_load_si512((const __m512i*)(outptr + 112));
                _sum8 = _mm512_load_si512((const __m512i*)(outptr + 128));
                _sum9 = _mm512_load_si512((const __m512i*)(outptr + 128 + 16));
                _suma = _mm512_load_si512((const __m512i*)(outptr + 128 + 32));
                _sumb = _mm512_load_si512((const __m512i*)(outptr + 128 + 48));
                _sumc = _mm512_load_si512((const __m512i*)(outptr + 128 + 64));
                _sumd = _mm512_load_si512((const __m512i*)(outptr + 128 + 80));
                _sume = _mm512_load_si512((const __m512i*)(outptr + 128 + 96));
                _sumf = _mm512_load_si512((const __m512i*)(outptr + 128 + 112));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512i _pA3 = _mm512_shuffle_epi32(_pA2, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_shuffle_i32x4(_pB0, _pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                _sum4 = _mm512_dpbusd_epi32(_sum4, _pB2, _pA0);
                _sum5 = _mm512_dpbusd_epi32(_sum5, _pB3, _pA0);
                _sum6 = _mm512_dpbusd_epi32(_sum6, _pB2, _pA1);
                _sum7 = _mm512_dpbusd_epi32(_sum7, _pB3, _pA1);
                _sum8 = _mm512_dpbusd_epi32(_sum8, _pB0, _pA2);
                _sum9 = _mm512_dpbusd_epi32(_sum9, _pB1, _pA2);
                _suma = _mm512_dpbusd_epi32(_suma, _pB0, _pA3);
                _sumb = _mm512_dpbusd_epi32(_sumb, _pB1, _pA3);
                _sumc = _mm512_dpbusd_epi32(_sumc, _pB2, _pA2);
                _sumd = _mm512_dpbusd_epi32(_sumd, _pB3, _pA2);
                _sume = _mm512_dpbusd_epi32(_sume, _pB2, _pA3);
                _sumf = _mm512_dpbusd_epi32(_sumf, _pB3, _pA3);
                pA += 64;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _w_shift1 = _mm512_shuffle_epi32(_w_shift0, _MM_PERM_BADC);
                __m512i _w_shift2 = _mm512_shuffle_i32x4(_w_shift0, _w_shift0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512i _w_shift3 = _mm512_shuffle_epi32(_w_shift2, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                _sum4 = _mm512_sub_epi32(_sum4, _w_shift0);
                _sum5 = _mm512_sub_epi32(_sum5, _w_shift0);
                _sum6 = _mm512_sub_epi32(_sum6, _w_shift1);
                _sum7 = _mm512_sub_epi32(_sum7, _w_shift1);
                _sum8 = _mm512_sub_epi32(_sum8, _w_shift2);
                _sum9 = _mm512_sub_epi32(_sum9, _w_shift2);
                _suma = _mm512_sub_epi32(_suma, _w_shift3);
                _sumb = _mm512_sub_epi32(_sumb, _w_shift3);
                _sumc = _mm512_sub_epi32(_sumc, _w_shift2);
                _sumd = _mm512_sub_epi32(_sumd, _w_shift2);
                _sume = _mm512_sub_epi32(_sume, _w_shift3);
                _sumf = _mm512_sub_epi32(_sumf, _w_shift3);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // 2301 6745 ab89 efcd
                // 4567 0123 cdef 89ab
                // 6745 2301 efcd ab89
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512i _pA3 = _mm512_shuffle_epi32(_pA2, _MM_PERM_BADC);

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                // 89ab cdef 0123 4567
                // 9ab8 defc 1230 5674
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_shuffle_i32x4(_pB0, _pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm512_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                _sum4 = _mm512_comp_dpwssd_epi32(_sum4, _pA0, _pB2);
                _sum5 = _mm512_comp_dpwssd_epi32(_sum5, _pA0, _pB3);
                _sum6 = _mm512_comp_dpwssd_epi32(_sum6, _pA1, _pB2);
                _sum7 = _mm512_comp_dpwssd_epi32(_sum7, _pA1, _pB3);
                _sum8 = _mm512_comp_dpwssd_epi32(_sum8, _pA2, _pB0);
                _sum9 = _mm512_comp_dpwssd_epi32(_sum9, _pA2, _pB1);
                _suma = _mm512_comp_dpwssd_epi32(_suma, _pA3, _pB0);
                _sumb = _mm512_comp_dpwssd_epi32(_sumb, _pA3, _pB1);
                _sumc = _mm512_comp_dpwssd_epi32(_sumc, _pA2, _pB2);
                _sumd = _mm512_comp_dpwssd_epi32(_sumd, _pA2, _pB3);
                _sume = _mm512_comp_dpwssd_epi32(_sume, _pA3, _pB2);
                _sumf = _mm512_comp_dpwssd_epi32(_sumf, _pA3, _pB3);

                pA += 32;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256i _pA2 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256i _pA3 = _mm256_shuffle_epi32(_pA2, _MM_SHUFFLE(2, 3, 0, 1));

                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0)));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1)));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2)));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3)));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB2)));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB3)));
                _sum8 = _mm512_add_epi32(_sum8, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB0)));
                _sum9 = _mm512_add_epi32(_sum9, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB1)));
                _suma = _mm512_add_epi32(_suma, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB0)));
                _sumb = _mm512_add_epi32(_sumb, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB1)));
                _sumc = _mm512_add_epi32(_sumc, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB2)));
                _sumd = _mm512_add_epi32(_sumd, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB3)));
                _sume = _mm512_add_epi32(_sume, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB2)));
                _sumf = _mm512_add_epi32(_sumf, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB3)));

                pA += 16;
                pB += 16;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            _mm512_store_si512((__m512i*)(outptr + 64), _sum4);
            _mm512_store_si512((__m512i*)(outptr + 80), _sum5);
            _mm512_store_si512((__m512i*)(outptr + 96), _sum6);
            _mm512_store_si512((__m512i*)(outptr + 112), _sum7);
            _mm512_store_si512((__m512i*)(outptr + 128), _sum8);
            _mm512_store_si512((__m512i*)(outptr + 128 + 16), _sum9);
            _mm512_store_si512((__m512i*)(outptr + 128 + 32), _suma);
            _mm512_store_si512((__m512i*)(outptr + 128 + 48), _sumb);
            _mm512_store_si512((__m512i*)(outptr + 128 + 64), _sumc);
            _mm512_store_si512((__m512i*)(outptr + 128 + 80), _sumd);
            _mm512_store_si512((__m512i*)(outptr + 128 + 96), _sume);
            _mm512_store_si512((__m512i*)(outptr + 128 + 112), _sumf);
            outptr += 256;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;
            __m512i _sum4;
            __m512i _sum5;
            __m512i _sum6;
            __m512i _sum7;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
                _sum4 = _mm512_setzero_si512();
                _sum5 = _mm512_setzero_si512();
                _sum6 = _mm512_setzero_si512();
                _sum7 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
                _sum4 = _mm512_load_si512((const __m512i*)(outptr + 64));
                _sum5 = _mm512_load_si512((const __m512i*)(outptr + 80));
                _sum6 = _mm512_load_si512((const __m512i*)(outptr + 96));
                _sum7 = _mm512_load_si512((const __m512i*)(outptr + 112));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                __m512i _pB0 = combine8x2_epi32(_pB, _pB);
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                _sum4 = _mm512_dpbusd_epi32(_sum4, _pB2, _pA0);
                _sum5 = _mm512_dpbusd_epi32(_sum5, _pB3, _pA0);
                _sum6 = _mm512_dpbusd_epi32(_sum6, _pB2, _pA1);
                _sum7 = _mm512_dpbusd_epi32(_sum7, _pB3, _pA1);
                pA += 64;
                pB += 32;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _w_shift1 = _mm512_shuffle_epi32(_w_shift0, _MM_PERM_BADC);
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // 2301 6745 ab89 efcd
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                // 0123 4567 0123 4567
                // 1230 5674 1230 5674
                // 4567 0123 4567 0123
                // 5674 1230 5674 1230
                __m512i _pB0 = combine8x2_epi32(_pBB, _pBB);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm512_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                _sum4 = _mm512_comp_dpwssd_epi32(_sum4, _pA0, _pB2);
                _sum5 = _mm512_comp_dpwssd_epi32(_sum5, _pA0, _pB3);
                _sum6 = _mm512_comp_dpwssd_epi32(_sum6, _pA1, _pB2);
                _sum7 = _mm512_comp_dpwssd_epi32(_sum7, _pA1, _pB3);

                pA += 32;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));

                __m256i _pB0 = combine4x2_epi32(_pB, _pB);
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0)));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1)));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2)));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3)));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB2)));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB3)));

                pA += 16;
                pB += 8;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            _mm512_store_si512((__m512i*)(outptr + 64), _sum4);
            _mm512_store_si512((__m512i*)(outptr + 80), _sum5);
            _mm512_store_si512((__m512i*)(outptr + 96), _sum6);
            _mm512_store_si512((__m512i*)(outptr + 112), _sum7);
            outptr += 128;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pB));
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                pA += 64;
                pB += 16;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _w_shift1 = _mm512_shuffle_epi32(_w_shift0, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // 2301 6745 ab89 efcd
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                // 0123 0123 0123 0123
                // 1230 1230 1230 1230
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm512_comp_dpwssd_epi32(_sum3, _pA1, _pB1);

                pA += 32;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));

                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0)));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1)));

                pA += 16;
                pB += 4;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            outptr += 64;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pB)[0]));
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CDAB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA);
                pA += 64;
                pB += 8;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift = _mm512_loadu_si512((const __m512i*)pA);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef

                // 0101 0101 0101 0101
                // 1010 1010 1010 1010
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CDAB);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);

                pA += 32;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));

                pA += 16;
                pB += 2;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            outptr += 32;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m512i _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB = _mm512_set1_epi32(((const int*)pB)[0]);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB, _pA);
                pA += 64;
                pB += 4;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift = _mm512_loadu_si512((const __m512i*)pA);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pBBBB = _mm512_cvtepi8_epi16(_pB);

                // 0xxx0xxx0xxx0xxx -> 00000000...
                __m512i _pB0 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_AAAA);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);

                pA += 32;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m256i _pB = _mm256_set1_epi16(pB[0]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB)));

                pA += 16;
                pB += 1;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            outptr += 16;
        }

        pAT += max_kk * 16;
#if __AVX512VNNI__
        if (max_kk >= 4)
        {
            pAT += 64;
        }
#endif // __AVX512VNNI__
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;
            __m512i _sum4;
            __m512i _sum5;
            __m512i _sum6;
            __m512i _sum7;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
                _sum4 = _mm512_setzero_si512();
                _sum5 = _mm512_setzero_si512();
                _sum6 = _mm512_setzero_si512();
                _sum7 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
                _sum4 = _mm512_load_si512((const __m512i*)(outptr + 64));
                _sum5 = _mm512_load_si512((const __m512i*)(outptr + 80));
                _sum6 = _mm512_load_si512((const __m512i*)(outptr + 96));
                _sum7 = _mm512_load_si512((const __m512i*)(outptr + 112));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA00 = combine8x2_epi32(_pA0, _pA0);
                __m512i _pA11 = _mm512_shuffle_epi32(_pA00, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA00);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA00);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA11);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA11);
                _sum4 = _mm512_dpbusd_epi32(_sum4, _pB2, _pA00);
                _sum5 = _mm512_dpbusd_epi32(_sum5, _pB3, _pA00);
                _sum6 = _mm512_dpbusd_epi32(_sum6, _pB2, _pA11);
                _sum7 = _mm512_dpbusd_epi32(_sum7, _pB3, _pA11);
                pA += 32;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                __m512i _w_shift00 = combine8x2_epi32(_w_shift0, _w_shift0);
                __m512i _w_shift11 = _mm512_shuffle_epi32(_w_shift00, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift00);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift00);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift11);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift11);
                _sum4 = _mm512_sub_epi32(_sum4, _w_shift00);
                _sum5 = _mm512_sub_epi32(_sum5, _w_shift00);
                _sum6 = _mm512_sub_epi32(_sum6, _w_shift11);
                _sum7 = _mm512_sub_epi32(_sum7, _w_shift11);
                pA += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 0123 4567
                // 2301 6745 2301 6745
                __m512i _pA00 = combine8x2_epi32(_pA0, _pA0);
                __m512i _pA11 = _mm512_shuffle_epi32(_pA00, _MM_PERM_BADC);

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                // 4567 0123 cdef 89ab
                // 5674 1230 defc 9ab8
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA00, _pB0);
                _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA00, _pB1);
                _sum2 = _mm512_comp_dpwssd_epi32(_sum2, _pA11, _pB0);
                _sum3 = _mm512_comp_dpwssd_epi32(_sum3, _pA11, _pB1);
                _sum4 = _mm512_comp_dpwssd_epi32(_sum4, _pA00, _pB2);
                _sum5 = _mm512_comp_dpwssd_epi32(_sum5, _pA00, _pB3);
                _sum6 = _mm512_comp_dpwssd_epi32(_sum6, _pA11, _pB2);
                _sum7 = _mm512_comp_dpwssd_epi32(_sum7, _pA11, _pB3);

                pA += 16;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                _pA = _mm_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                __m256i _pA00 = combine4x2_epi32(_pA, _pA);
                __m256i _pA11 = _mm256_shuffle_epi32(_pA00, _MM_SHUFFLE(2, 3, 0, 1));

                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB0)));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB1)));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB0)));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB1)));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB2)));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB3)));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB2)));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB3)));

                pA += 8;
                pB += 16;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            _mm512_store_si512((__m512i*)(outptr + 64), _sum4);
            _mm512_store_si512((__m512i*)(outptr + 80), _sum5);
            _mm512_store_si512((__m512i*)(outptr + 96), _sum6);
            _mm512_store_si512((__m512i*)(outptr + 112), _sum7);

            outptr += 128;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;
            __m256i _sum4;
            __m256i _sum5;
            __m256i _sum6;
            __m256i _sum7;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
                _sum2 = _mm256_setzero_si256();
                _sum3 = _mm256_setzero_si256();
                _sum4 = _mm256_setzero_si256();
                _sum5 = _mm256_setzero_si256();
                _sum6 = _mm256_setzero_si256();
                _sum7 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
                _sum4 = _mm256_load_si256((const __m256i*)(outptr + 32));
                _sum5 = _mm256_load_si256((const __m256i*)(outptr + 40));
                _sum6 = _mm256_load_si256((const __m256i*)(outptr + 48));
                _sum7 = _mm256_load_si256((const __m256i*)(outptr + 56));
            }

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pB);
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pB2, _MM_SHUFFLE(0, 3, 2, 1));
#if __AVXVNNIINT8__
                _sum0 = _mm256_dpbssd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm256_dpbssd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm256_dpbssd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm256_dpbssd_epi32(_sum3, _pB1, _pA1);
                _sum4 = _mm256_dpbssd_epi32(_sum4, _pB2, _pA0);
                _sum5 = _mm256_dpbssd_epi32(_sum5, _pB3, _pA0);
                _sum6 = _mm256_dpbssd_epi32(_sum6, _pB2, _pA1);
                _sum7 = _mm256_dpbssd_epi32(_sum7, _pB3, _pA1);
#else  // __AVXVNNIINT8__
                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _pB1, _pA1);
                _sum4 = _mm256_comp_dpbusd_epi32(_sum4, _pB2, _pA0);
                _sum5 = _mm256_comp_dpbusd_epi32(_sum5, _pB3, _pA0);
                _sum6 = _mm256_comp_dpbusd_epi32(_sum6, _pB2, _pA1);
                _sum7 = _mm256_comp_dpbusd_epi32(_sum7, _pB3, _pA1);
#endif // __AVXVNNIINT8__
                pA += 32;
                pB += 32;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _w_shift1 = _mm256_shuffle_epi32(_w_shift0, _MM_SHUFFLE(1, 0, 3, 2));
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
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX512F__
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
#else
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
#endif
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567
                // 2301 6745
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 1230 5674
                // 4567 0123
                // 5674 1230
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pB2, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm256_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm256_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                _sum4 = _mm256_comp_dpwssd_epi32(_sum4, _pA0, _pB2);
                _sum5 = _mm256_comp_dpwssd_epi32(_sum5, _pA0, _pB3);
                _sum6 = _mm256_comp_dpwssd_epi32(_sum6, _pA1, _pB2);
                _sum7 = _mm256_comp_dpwssd_epi32(_sum7, _pA1, _pB3);

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA0 = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB0 = _mm_loadl_epi64((const __m128i*)pB);

                _pA0 = _mm_cvtepi8_epi16(_pA0);
                _pB0 = _mm_cvtepi8_epi16(_pB0);

                __m128i _pA1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pA0, _MM_SHUFFLE(1, 0, 3, 2)), _MM_SHUFFLE(1, 0, 3, 2));
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB2 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128i _pB3 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0)));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1)));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0)));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1)));
                _sum4 = _mm256_add_epi32(_sum4, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB2)));
                _sum5 = _mm256_add_epi32(_sum5, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB3)));
                _sum6 = _mm256_add_epi32(_sum6, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB2)));
                _sum7 = _mm256_add_epi32(_sum7, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB3)));

                pA += 8;
                pB += 8;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);
            _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
            _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
            _mm256_store_si256((__m256i*)(outptr + 24), _sum3);
            _mm256_store_si256((__m256i*)(outptr + 32), _sum4);
            _mm256_store_si256((__m256i*)(outptr + 40), _sum5);
            _mm256_store_si256((__m256i*)(outptr + 48), _sum6);
            _mm256_store_si256((__m256i*)(outptr + 56), _sum7);

            outptr += 64;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
                _sum2 = _mm256_setzero_si256();
                _sum3 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
            }

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                __m256i _pB0 = combine4x2_epi32(_pB, _pB);
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
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
                pA += 32;
                pB += 16;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _w_shift1 = _mm256_shuffle_epi32(_w_shift0, _MM_SHUFFLE(1, 0, 3, 2));
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm256_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm256_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm256_sub_epi32(_sum3, _w_shift1);
                pA += 32;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX512F__
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
#else
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
#endif
                __m128i _pB = _mm_castpd_si128(_mm_load1_pd((const double*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567
                // 2301 6745
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 0123
                // 1230 1230
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm256_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm256_comp_dpwssd_epi32(_sum3, _pA1, _pB1);

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
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

                pA += 8;
                pB += 4;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);
            _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
            _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
            _mm256_store_si256((__m256i*)(outptr + 24), _sum3);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
            }

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB0 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 1, 0, 1));
#if __AVXVNNIINT8__
                _sum0 = _mm256_dpbssd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm256_dpbssd_epi32(_sum1, _pB1, _pA);
#else  // __AVXVNNIINT8__
                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB1, _pA);
#endif // __AVXVNNIINT8__
                pA += 32;
                pB += 8;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m256i _w_shift = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm256_sub_epi32(_sum1, _w_shift);
                pA += 32;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX512F__
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
#else
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
#endif
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567

                // 0101 0101
                // 1010 1010
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 1, 0, 1));

                _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA0, _pB1);

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB0 = _mm_set1_epi16(((const short*)pB)[0]);

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB0 = _mm_cvtepi8_epi16(_pB0);

                // 01234567

                // 01010101
                // 10101010
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0)));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1)));

                pA += 8;
                pB += 2;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);
            _mm256_store_si256((__m256i*)(outptr + 8), _sum1);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m256i _sum0;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
            }

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));
#if __AVXVNNIINT8__
                _sum0 = _mm256_dpbssd_epi32(_sum0, _pB, _pA);
#else // __AVXVNNIINT8__
#if __AVX512VNNI__ && _MSC_VER < 1932
                // old msvc crash here  --- nihui
                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepu8_epi16(_pB);
                __m512i _s0 = _mm512_madd_epi16(_pA0, _pB0);
                __m256i _s1 = _mm256_hadd_epi32(_mm512_extracti32x8_epi32(_s0, 0), _mm512_extracti32x8_epi32(_s0, 1));
                _sum0 = _mm256_add_epi32(_sum0, _mm256_permute4x64_epi64(_s1, _MM_SHUFFLE(3, 1, 2, 0)));
#else
                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB, _pA);
#endif
#endif // __AVXVNNIINT8__
                pA += 32;
                pB += 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m256i _w_shift = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift);
                pA += 32;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX512F__
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
#else
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
#endif
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

                // 0xxx0xxx -> 00000000 11111111
                __m256i _pB0 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(0, 0, 0, 0));

                _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);

                pA += 16;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(pB[0]);

                _pA = _mm_cvtepi8_epi16(_pA);

                _sum0 = _mm256_add_epi32(_sum0, _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB)));

                pA += 8;
                pB += 1;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);

            outptr += 8;
        }

        pAT += max_kk * 8;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        if (max_kk >= 4)
        {
            pAT += 32;
        }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
                _sum1 = _mm512_loadu_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_loadu_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_loadu_si512((const __m512i*)(outptr + 48));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pA));
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                pA += 16;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pA));
                __m512i _w_shift1 = _mm512_shuffle_epi32(_w_shift0, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                pA += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pA));
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 0123 0123 0123
                // 2301 2301 2301 2301
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm512_comp_dpwssd_epi32(_sum3, _pA1, _pB1);

                pA += 8;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));

                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0)));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1)));

                pA += 4;
                pB += 16;
            }

            _mm512_storeu_si512((__m512i*)outptr, _sum0);
            _mm512_storeu_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_storeu_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_storeu_si512((__m512i*)(outptr + 48), _sum3);

            outptr += 64;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if __AVX2__
            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;
#else  // __AVX2__
            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;
            __m128i _sum4;
            __m128i _sum5;
            __m128i _sum6;
            __m128i _sum7;
#endif // __AVX2__

            if (k == 0)
            {
#if __AVX2__
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
                _sum2 = _mm256_setzero_si256();
                _sum3 = _mm256_setzero_si256();
#else  // __AVX2__
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
                _sum4 = _mm_setzero_si128();
                _sum5 = _mm_setzero_si128();
                _sum6 = _mm_setzero_si128();
                _sum7 = _mm_setzero_si128();
#endif // __AVX2__
            }
            else
            {
#if __AVX2__
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
#else  // __AVX2__
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
                _sum4 = _mm_load_si128((const __m128i*)(outptr + 16));
                _sum5 = _mm_load_si128((const __m128i*)(outptr + 20));
                _sum6 = _mm_load_si128((const __m128i*)(outptr + 24));
                _sum7 = _mm_load_si128((const __m128i*)(outptr + 28));
#endif // __AVX2__
            }

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                __m256i _pB01 = _mm256_loadu_si256((const __m256i*)pB);
                __m256i _pA00 = combine4x2_epi32(_pA0, _pA0);
                __m256i _pA11 = _mm256_shuffle_epi32(_pA00, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB23 = _mm256_shuffle_epi32(_pB01, _MM_SHUFFLE(0, 3, 2, 1));
#if __AVXVNNIINT8__
                _sum0 = _mm256_dpbssd_epi32(_sum0, _pB01, _pA00);
                _sum1 = _mm256_dpbssd_epi32(_sum1, _pB01, _pA11);
                _sum2 = _mm256_dpbssd_epi32(_sum2, _pB23, _pA00);
                _sum3 = _mm256_dpbssd_epi32(_sum3, _pB23, _pA11);
#else  // __AVXVNNIINT8__
                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB01, _pA00);
                _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB01, _pA11);
                _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _pB23, _pA00);
                _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _pB23, _pA11);
#endif // __AVXVNNIINT8__
                pA += 16;
                pB += 32;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m128i _w_shift0 = _mm_loadu_si128((const __m128i*)pA);
                __m256i _w_shift00 = combine4x2_epi32(_w_shift0, _w_shift0);
                __m256i _w_shift11 = _mm256_shuffle_epi32(_w_shift00, _MM_SHUFFLE(1, 0, 3, 2));
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift00);
                _sum1 = _mm256_sub_epi32(_sum1, _w_shift11);
                _sum2 = _mm256_sub_epi32(_sum2, _w_shift00);
                _sum3 = _mm256_sub_epi32(_sum3, _w_shift11);
                pA += 16;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __AVX2__
                __m256i _pA00 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB01 = _mm256_cvtepi8_epi16(_pB);

                __m256i _pA11 = _mm256_shuffle_epi32(_pA00, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB23 = _mm256_shuffle_epi32(_pB01, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA00, _pB01);
                _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA11, _pB01);
                _sum2 = _mm256_comp_dpwssd_epi32(_sum2, _pA00, _pB23);
                _sum3 = _mm256_comp_dpwssd_epi32(_sum3, _pA11, _pB23);
#else // __AVX2__
#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif
                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pBl = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pBh = _mm_unpackhi_epi8(_pB, _extpB);

                // 0123
                // 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123
                // 4567
                // 1230
                // 5674
                __m128i _pB0 = _pBl;
                __m128i _pB1 = _pBh;
                __m128i _pB2 = _mm_shuffle_epi32(_pBl, _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB3 = _mm_shuffle_epi32(_pBh, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
                _sum4 = _mm_comp_dpwssd_epi32(_sum4, _pA0, _pB2);
                _sum5 = _mm_comp_dpwssd_epi32(_sum5, _pA0, _pB3);
                _sum6 = _mm_comp_dpwssd_epi32(_sum6, _pA1, _pB2);
                _sum7 = _mm_comp_dpwssd_epi32(_sum7, _pA1, _pB3);
#endif // __AVX2__
                pA += 8;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __AVX2__
                // 01230123
                // 23012301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567
                // 12305674
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);
#else // __AVX2__
#if __XOP__
                // 00112233
                // 22330011
                __m128i _pA0 = _mm_unpacklo_epi16(_pA, _pA);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 00112233
                // 44556677
                // 1.2.3.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_unpackhi_epi16(_pB, _pB);
                __m128i _pB2 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB3 = _mm_shuffle_epi32(_pB1, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_maccd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maccd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maccd_epi16(_pA1, _pB1, _sum3);
                _sum4 = _mm_maccd_epi16(_pA0, _pB2, _sum4);
                _sum5 = _mm_maccd_epi16(_pA0, _pB3, _sum5);
                _sum6 = _mm_maccd_epi16(_pA1, _pB2, _sum6);
                _sum7 = _mm_maccd_epi16(_pA1, _pB3, _sum7);
#else
                // 01230123
                // 23012301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567
                // 12305674
                __m128i _pB01 = _pB;
                __m128i _pB23 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB01);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB01);
                __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB01);
                __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB01);
                __m128i _sl2 = _mm_mullo_epi16(_pA0, _pB23);
                __m128i _sh2 = _mm_mulhi_epi16(_pA0, _pB23);
                __m128i _sl3 = _mm_mullo_epi16(_pA1, _pB23);
                __m128i _sh3 = _mm_mulhi_epi16(_pA1, _pB23);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);
                __m128i _s4 = _mm_unpacklo_epi16(_sl2, _sh2);
                __m128i _s5 = _mm_unpackhi_epi16(_sl2, _sh2);
                __m128i _s6 = _mm_unpacklo_epi16(_sl3, _sh3);
                __m128i _s7 = _mm_unpackhi_epi16(_sl3, _sh3);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);
                _sum4 = _mm_add_epi32(_sum4, _s4);
                _sum5 = _mm_add_epi32(_sum5, _s5);
                _sum6 = _mm_add_epi32(_sum6, _s6);
                _sum7 = _mm_add_epi32(_sum7, _s7);
#endif // __XOP__
#endif // __AVX2__

                pA += 4;
                pB += 8;
            }

#if __AVX2__
            _mm256_store_si256((__m256i*)outptr, _sum0);
            _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
            _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
            _mm256_store_si256((__m256i*)(outptr + 24), _sum3);
#else  // __AVX2__
            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            _mm_store_si128((__m128i*)(outptr + 8), _sum2);
            _mm_store_si128((__m128i*)(outptr + 12), _sum3);
            _mm_store_si128((__m128i*)(outptr + 16), _sum4);
            _mm_store_si128((__m128i*)(outptr + 20), _sum5);
            _mm_store_si128((__m128i*)(outptr + 24), _sum6);
            _mm_store_si128((__m128i*)(outptr + 28), _sum7);
#endif // __AVX2__

            outptr += 32;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
            }

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
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
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m128i _w_shift0 = _mm_loadu_si128((const __m128i*)pA);
                __m128i _w_shift1 = _mm_shuffle_epi32(_w_shift0, _MM_SHUFFLE(1, 0, 3, 2));
                _sum0 = _mm_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm_sub_epi32(_sum3, _w_shift1);
                pA += 16;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0123
                // 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123
                // 1230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm_comp_dpwssd_epi32(_sum3, _pA1, _pB1);

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                // 00112233
                // 22330011
                __m128i _pA0 = _mm_unpacklo_epi16(_pA, _pA);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 00112233
                // 1.2.3.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_maccd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maccd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maccd_epi16(_pA1, _pB1, _sum3);
#else
                // 0123 0123
                // 2301 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 0123 1230
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB01);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB01);
                __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB01);
                __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB01);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);
#endif

                pA += 4;
                pB += 4;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            _mm_store_si128((__m128i*)(outptr + 8), _sum2);
            _mm_store_si128((__m128i*)(outptr + 12), _sum3);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
            }

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB0 = _mm_castpd_si128(_mm_load1_pd((const double*)pB));
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
#if __AVXVNNIINT8__
                _sum0 = _mm_dpbssd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm_dpbssd_epi32(_sum1, _pB1, _pA);
#else  // __AVXVNNIINT8__
                _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm_comp_dpbusd_epi32(_sum1, _pB1, _pA);
#endif // __AVXVNNIINT8__
                pA += 16;
                pB += 8;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift);
                pA += 16;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB0 = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB0 = _mm_cvtepi8_epi16(_pB0);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB0 = _mm_unpacklo_epi8(_pB0, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB0));
#endif

                // 0123

                // 0101
                // 1010
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA, _pB0);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA, _pB1);

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                // 00112233
                _pA = _mm_unpacklo_epi16(_pA, _pA);

                // 00110011
                // 1.0.1.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));

                _sum0 = _mm_maccd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA, _pB1, _sum1);
#else
                // 01230123
                // 01011010
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 1, 0, 1));

                __m128i _sl = _mm_mullo_epi16(_pA, _pB01);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB01);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
#endif

                pA += 4;
                pB += 2;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
            }

            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));
#if __AVXVNNIINT8__
                _sum0 = _mm_dpbssd_epi32(_sum0, _pB, _pA);
#else // __AVXVNNIINT8__
#if __AVX512VNNI__ && _MSC_VER < 1932
                // old msvc crash here  --- nihui
                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepu8_epi16(_pB);
                __m256i _s0 = _mm256_madd_epi16(_pA0, _pB0);
                __m128i _s1 = _mm_hadd_epi32(_mm256_extracti128_si256(_s0, 0), _mm256_extracti128_si256(_s0, 1));
                _sum0 = _mm_add_epi32(_sum0, _s1);
#else
                _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pB, _pA);
#endif
#endif // __AVXVNNIINT8__
                pA += 16;
                pB += 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                pA += 16;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA, _pB);

                pA += 8;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(pB[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

#if __XOP__
                _pA = _mm_unpacklo_epi16(_pA, _pA);

                _sum0 = _mm_maccd_epi16(_pA, _pB, _sum0);
#else
                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
#endif

                pA += 4;
                pB += 1;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);

            outptr += 4;
        }

        pAT += max_kk * 4;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        if (max_kk >= 4)
        {
            pAT += 16;
        }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _sum0;
            __m512i _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
                _sum1 = _mm512_loadu_si512((const __m512i*)(outptr + 16));
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pA)[0]));
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA);
                pA += 8;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pA)[0]));
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift);
                pA += 8;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pA));
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0101 0101 0101 0101

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_comp_dpwssd_epi32(_sum1, _pA0, _pB1);

                pA += 4;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01010101 01010101

                // 01234567 89abcdef
                // 12305674 9ab8defc
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0)));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1)));

                pA += 2;
                pB += 16;
            }

            _mm512_storeu_si512((__m512i*)outptr, _sum0);
            _mm512_storeu_si512((__m512i*)(outptr + 16), _sum1);

            outptr += 32;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX2__
            __m256i _sum0;
            __m256i _sum1;
#else
            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;
#endif

            if (k == 0)
            {
#if __AVX2__
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
#else
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
#endif
            }
            else
            {
#if __AVX2__
                _sum0 = _mm256_loadu_si256((const __m256i*)outptr);
                _sum1 = _mm256_loadu_si256((const __m256i*)(outptr + 8));
#else
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
#endif
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA00 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pA));
                __m256i _pA11 = _mm256_shuffle_epi32(_pA00, _MM_SHUFFLE(2, 3, 0, 1));
                __m256i _pB01 = _mm256_loadu_si256((const __m256i*)pB);
#if __AVXVNNIINT8__
                _sum0 = _mm256_dpbssd_epi32(_sum0, _pB01, _pA00);
                _sum1 = _mm256_dpbssd_epi32(_sum1, _pB01, _pA11);
#else  // __AVXVNNIINT8__
                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB01, _pA00);
                _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _pB01, _pA11);
#endif // __AVXVNNIINT8__
                pA += 8;
                pB += 32;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m256i _w_shift00 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pA));
                __m256i _w_shift11 = _mm256_shuffle_epi32(_w_shift00, _MM_SHUFFLE(2, 3, 0, 1));
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift00);
                _sum1 = _mm256_sub_epi32(_sum1, _w_shift11);
                pA += 8;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __AVX2__
                __m256i _pA00 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB01 = _mm256_cvtepi8_epi16(_pB);

                __m256i _pA11 = _mm256_shuffle_epi32(_pA00, _MM_SHUFFLE(2, 3, 0, 1));

                _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA00, _pB01);
                _sum1 = _mm256_comp_dpwssd_epi32(_sum1, _pA11, _pB01);
#else // __AVX2__
#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pB0 = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pB1 = _mm_unpackhi_epi8(_pB, _extpB);

                // 0101
                // 1010
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 0123
                // 4567

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm_comp_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm_comp_dpwssd_epi32(_sum3, _pA1, _pB1);
#endif // __AVX2__

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 01010101
                // 10101010
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pA, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567

#if __AVX2__
                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
#else  // __AVX2__
                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB);
                __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB);
                __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);
#endif // __AVX2__

                pA += 2;
                pB += 8;
            }

#if __AVX2__
            _mm256_storeu_si256((__m256i*)outptr, _sum0);
            _mm256_storeu_si256((__m256i*)(outptr + 8), _sum1);
#else
            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            _mm_store_si128((__m128i*)(outptr + 8), _sum2);
            _mm_store_si128((__m128i*)(outptr + 12), _sum3);
#endif

            outptr += 16;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
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
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift);
                pA += 8;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0101

                // 0123
                // 1230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA, _pB0);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA, _pB1);

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 01010101

                // 01231230
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl = _mm_mullo_epi16(_pA, _pB01);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB01);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);

                pA += 2;
                pB += 4;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);

            outptr += 8;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum00;
            int sum01;
            int sum10;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum01 = 0;
                sum10 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVXVNNIINT8__
                sum00 += pA[0] * pB[0];
                sum00 += pA[1] * pB[1];
                sum00 += pA[2] * pB[2];
                sum00 += pA[3] * pB[3];
                sum01 += pA[0] * pB[4];
                sum01 += pA[1] * pB[5];
                sum01 += pA[2] * pB[6];
                sum01 += pA[3] * pB[7];
                sum10 += pA[4] * pB[0];
                sum10 += pA[5] * pB[1];
                sum10 += pA[6] * pB[2];
                sum10 += pA[7] * pB[3];
                sum11 += pA[4] * pB[4];
                sum11 += pA[5] * pB[5];
                sum11 += pA[6] * pB[6];
                sum11 += pA[7] * pB[7];
#else  // __AVXVNNIINT8__
                sum00 += pA[0] * ((unsigned char*)pB)[0];
                sum00 += pA[1] * ((unsigned char*)pB)[1];
                sum00 += pA[2] * ((unsigned char*)pB)[2];
                sum00 += pA[3] * ((unsigned char*)pB)[3];
                sum01 += pA[0] * ((unsigned char*)pB)[4];
                sum01 += pA[1] * ((unsigned char*)pB)[5];
                sum01 += pA[2] * ((unsigned char*)pB)[6];
                sum01 += pA[3] * ((unsigned char*)pB)[7];
                sum10 += pA[4] * ((unsigned char*)pB)[0];
                sum10 += pA[5] * ((unsigned char*)pB)[1];
                sum10 += pA[6] * ((unsigned char*)pB)[2];
                sum10 += pA[7] * ((unsigned char*)pB)[3];
                sum11 += pA[4] * ((unsigned char*)pB)[4];
                sum11 += pA[5] * ((unsigned char*)pB)[5];
                sum11 += pA[6] * ((unsigned char*)pB)[6];
                sum11 += pA[7] * ((unsigned char*)pB)[7];
#endif // __AVXVNNIINT8__
                pA += 8;
                pB += 8;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                int w_shift0 = ((int*)pA)[0];
                int w_shift1 = ((int*)pA)[1];
                sum00 = sum00 - w_shift0;
                sum01 = sum01 - w_shift0;
                sum10 = sum10 - w_shift1;
                sum11 = sum11 - w_shift1;
                pA += 8;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum00 += pA[0] * pB[0];
                sum00 += pA[1] * pB[1];
                sum01 += pA[0] * pB[2];
                sum01 += pA[1] * pB[3];
                sum10 += pA[2] * pB[0];
                sum10 += pA[3] * pB[1];
                sum11 += pA[2] * pB[2];
                sum11 += pA[3] * pB[3];
                pA += 4;
                pB += 4;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[0] * pB[1];
                sum10 += pA[1] * pB[0];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVXVNNIINT8__
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum0 += pA[2] * pB[2];
                sum0 += pA[3] * pB[3];
                sum1 += pA[4] * pB[0];
                sum1 += pA[5] * pB[1];
                sum1 += pA[6] * pB[2];
                sum1 += pA[7] * pB[3];
#else  // __AVXVNNIINT8__
                sum0 += pA[0] * ((unsigned char*)pB)[0];
                sum0 += pA[1] * ((unsigned char*)pB)[1];
                sum0 += pA[2] * ((unsigned char*)pB)[2];
                sum0 += pA[3] * ((unsigned char*)pB)[3];
                sum1 += pA[4] * ((unsigned char*)pB)[0];
                sum1 += pA[5] * ((unsigned char*)pB)[1];
                sum1 += pA[6] * ((unsigned char*)pB)[2];
                sum1 += pA[7] * ((unsigned char*)pB)[3];
#endif // __AVXVNNIINT8__
                pA += 8;
                pB += 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                int w_shift0 = ((int*)pA)[0];
                int w_shift1 = ((int*)pA)[1];
                sum0 = sum0 - w_shift0;
                sum1 = sum1 - w_shift1;
                pA += 8;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[2] * pB[0];
                sum1 += pA[3] * pB[1];
                pA += 4;
                pB += 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        pAT += max_kk * 2;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        if (max_kk >= 4)
        {
            pAT += 8;
        }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA = _mm512_set1_epi32(((const int*)pA)[0]);
                __m512i _pB = _mm512_loadu_si512((const __m512i*)pB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB, _pA);
                pA += 4;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift = _mm512_set1_epi32(((const int*)pA)[0]);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift);
                pA += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_set1_epi16(((const short*)pA)[0]);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                _sum0 = _mm512_comp_dpwssd_epi32(_sum0, _pA0, _pB0);

                pA += 2;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = _mm256_set1_epi16(pA[0]);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA, _pB0)));

                pA += 1;
                pB += 16;
            }

            _mm512_storeu_si512((__m512i*)outptr, _sum0);

            outptr += 16;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX2__
            __m256i _sum0;
#else
            __m128i _sum0;
            __m128i _sum1;
#endif

            if (k == 0)
            {
#if __AVX2__
                _sum0 = _mm256_setzero_si256();
#else
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
#endif
            }
            else
            {
#if __AVX2__
                _sum0 = _mm256_loadu_si256((const __m256i*)outptr);
#else
                _sum0 = _mm_loadu_si128((const __m128i*)outptr);
                _sum1 = _mm_loadu_si128((const __m128i*)(outptr + 4));
#endif
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA00 = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pA));
                __m256i _pB01 = _mm256_loadu_si256((const __m256i*)pB);
#if __AVXVNNIINT8__
                _sum0 = _mm256_dpbssd_epi32(_sum0, _pB01, _pA00);
#else  // __AVXVNNIINT8__
                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _pB01, _pA00);
#endif // __AVXVNNIINT8__
                pA += 4;
                pB += 32;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m256i _w_shift = _mm256_set1_epi32(((const int*)pA)[0]);
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift);
                pA += 4;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __AVX2__
                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                _sum0 = _mm256_comp_dpwssd_epi32(_sum0, _pA0, _pB0);
#else
#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pB0 = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pB1 = _mm_unpackhi_epi8(_pB, _extpB);

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA, _pB0);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _pA, _pB1);
#endif // __AVX2__

                pA += 2;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(pA[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __AVX2__
                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
#else
                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
#endif // __AVX2__

                pA += 1;
                pB += 8;
            }

#if __AVX2__
            _mm256_storeu_si256((__m256i*)outptr, _sum0);
#else
            _mm_storeu_si128((__m128i*)outptr, _sum0);
            _mm_storeu_si128((__m128i*)(outptr + 4), _sum1);
#endif

            outptr += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_loadu_si128((const __m128i*)outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
#if __AVXVNNIINT8__
                _sum0 = _mm_dpbssd_epi32(_sum0, _pB, _pA);
#else  // __AVXVNNIINT8__
                _sum0 = _mm_comp_dpbusd_epi32(_sum0, _pB, _pA);
#endif // __AVXVNNIINT8__
                pA += 4;
                pB += 16;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_set1_epi32(((const int*)pA)[0]);
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                pA += 4;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0xxx -> 0000
                __m128i _pA0 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 0, 0, 0));

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _pA0, _pB);

                pA += 2;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(pA[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);

                pA += 1;
                pB += 4;
            }

            _mm_storeu_si128((__m128i*)outptr, _sum0);

            outptr += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVXVNNIINT8__
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum0 += pA[2] * pB[2];
                sum0 += pA[3] * pB[3];
                sum1 += pA[0] * pB[4];
                sum1 += pA[1] * pB[5];
                sum1 += pA[2] * pB[6];
                sum1 += pA[3] * pB[7];
#else  // __AVXVNNIINT8__
                sum0 += pA[0] * ((unsigned char*)pB)[0];
                sum0 += pA[1] * ((unsigned char*)pB)[1];
                sum0 += pA[2] * ((unsigned char*)pB)[2];
                sum0 += pA[3] * ((unsigned char*)pB)[3];
                sum1 += pA[0] * ((unsigned char*)pB)[4];
                sum1 += pA[1] * ((unsigned char*)pB)[5];
                sum1 += pA[2] * ((unsigned char*)pB)[6];
                sum1 += pA[3] * ((unsigned char*)pB)[7];
#endif // __AVXVNNIINT8__
                pA += 4;
                pB += 8;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                int w_shift = ((const int*)pA)[0];
                sum0 = sum0 - w_shift;
                sum1 = sum1 - w_shift;
                pA += 4;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[0] * pB[2];
                sum1 += pA[1] * pB[3];
                pA += 2;
                pB += 4;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                pA += 1;
                pB += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum;

            if (k == 0)
            {
                sum = 0;
            }
            else
            {
                sum = outptr[0];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__ || __AVXVNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVXVNNIINT8__
                sum += pA[0] * pB[0];
                sum += pA[1] * pB[1];
                sum += pA[2] * pB[2];
                sum += pA[3] * pB[3];
#else  // __AVXVNNIINT8__
                sum += pA[0] * ((unsigned char*)pB)[0];
                sum += pA[1] * ((unsigned char*)pB)[1];
                sum += pA[2] * ((unsigned char*)pB)[2];
                sum += pA[3] * ((unsigned char*)pB)[3];
#endif // __AVXVNNIINT8__
                pA += 4;
                pB += 4;
            }
#if !__AVXVNNIINT8__
            if (max_kk >= 4)
            {
                int w_shift = ((const int*)pA)[0];
                sum = sum - w_shift;
                pA += 4;
            }
#endif // !__AVXVNNIINT8__
#endif // __AVX512VNNI__ || __AVXVNNI__
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            outptr[0] = sum;

            outptr += 1;
        }

        pAT += max_kk;
#if __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
        if (max_kk >= 4)
        {
            pAT += 4;
        }
#endif // __AVX512VNNI__ || (__AVXVNNI__ && !__AVXVNNIINT8__)
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(int)));

#if __AVX512F__
    TILE_M = std::max(16, tile_size / 16 * 16);
    TILE_N = std::max(16, tile_size / 16 * 16);
    TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX__
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(1, tile_size);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

    if (K > 0)
    {
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

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

#if __AVX512F__
            TILE_M = std::max(16, tile_size / 16 * 16);
            TILE_N = std::max(16, tile_size / 16 * 16);
#elif __AVX__
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
            TILE_N = std::max(1, tile_size);
#endif
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
#endif
    }

    if (N > 0)
    {
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
#endif
    }

    if (constant_TILE_N > 0)
    {
#if __AVX512F__
        TILE_N = (constant_TILE_N + 15) / 16 * 16;
#elif __AVX__
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#elif __SSE2__
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = constant_TILE_N;
#endif
    }

    if (constant_TILE_K > 0)
    {
#if __AVX512F__
        TILE_K = (constant_TILE_K + 15) / 16 * 16;
#elif __AVX__
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#elif __SSE2__
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

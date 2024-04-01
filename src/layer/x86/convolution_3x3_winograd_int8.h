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

#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void conv3x3s1_winograd23_int8_avx512vnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt);
void conv3x3s1_winograd43_int8_avx512vnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
void conv3x3s1_winograd23_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt);
void conv3x3s1_winograd43_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void conv3x3s1_winograd23_transform_kernel_int8_avx2(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt);
void conv3x3s1_winograd23_int8_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt);
void conv3x3s1_winograd43_transform_kernel_int8_avx2(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt);
void conv3x3s1_winograd43_int8_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
void conv3x3s1_winograd23_int8_xop(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt);
void conv3x3s1_winograd43_int8_xop(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt);
#endif
#endif

static void pack_A_tile_int8(const Mat& A, Mat& AT, int batch, int max_ii, int max_kk)
{
    const int N = max_kk * batch;

    for (int b = 0; b < batch; b++)
    {
        short* pp = AT.row<short>(b);

        int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[batch];
                pp[2] = p0[N];
                pp[3] = p0[N + batch];
                pp[4] = p0[N * 2];
                pp[5] = p0[N * 2 + batch];
                pp[6] = p0[N * 3];
                pp[7] = p0[N * 3 + batch];
                pp[8] = p0[N * 4];
                pp[9] = p0[N * 4 + batch];
                pp[10] = p0[N * 5];
                pp[11] = p0[N * 5 + batch];
                pp[12] = p0[N * 6];
                pp[13] = p0[N * 6 + batch];
                pp[14] = p0[N * 7];
                pp[15] = p0[N * 7 + batch];
                pp[16] = p0[N * 8];
                pp[17] = p0[N * 8 + batch];
                pp[18] = p0[N * 9];
                pp[19] = p0[N * 9 + batch];
                pp[20] = p0[N * 10];
                pp[21] = p0[N * 10 + batch];
                pp[22] = p0[N * 11];
                pp[23] = p0[N * 11 + batch];
                pp[24] = p0[N * 12];
                pp[25] = p0[N * 12 + batch];
                pp[26] = p0[N * 13];
                pp[27] = p0[N * 13 + batch];
                pp[28] = p0[N * 14];
                pp[29] = p0[N * 14 + batch];
                pp[30] = p0[N * 15];
                pp[31] = p0[N * 15 + batch];
                p0 += batch * 2;
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[N * 2];
                pp[3] = p0[N * 3];
                pp[4] = p0[N * 4];
                pp[5] = p0[N * 5];
                pp[6] = p0[N * 6];
                pp[7] = p0[N * 7];
                pp[8] = p0[N * 8];
                pp[9] = p0[N * 9];
                pp[10] = p0[N * 10];
                pp[11] = p0[N * 11];
                pp[12] = p0[N * 12];
                pp[13] = p0[N * 13];
                pp[14] = p0[N * 14];
                pp[15] = p0[N * 15];
                p0 += batch;
                pp += 16;
            }
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[batch];
                pp[2] = p0[N];
                pp[3] = p0[N + batch];
                pp[4] = p0[N * 2];
                pp[5] = p0[N * 2 + batch];
                pp[6] = p0[N * 3];
                pp[7] = p0[N * 3 + batch];
                pp[8] = p0[N * 4];
                pp[9] = p0[N * 4 + batch];
                pp[10] = p0[N * 5];
                pp[11] = p0[N * 5 + batch];
                pp[12] = p0[N * 6];
                pp[13] = p0[N * 6 + batch];
                pp[14] = p0[N * 7];
                pp[15] = p0[N * 7 + batch];
                p0 += batch * 2;
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[N * 2];
                pp[3] = p0[N * 3];
                pp[4] = p0[N * 4];
                pp[5] = p0[N * 5];
                pp[6] = p0[N * 6];
                pp[7] = p0[N * 7];
                p0 += batch;
                pp += 8;
            }
        }
#endif // __AVX2__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[batch];
                pp[2] = p0[N];
                pp[3] = p0[N + batch];
                pp[4] = p0[N * 2];
                pp[5] = p0[N * 2 + batch];
                pp[6] = p0[N * 3];
                pp[7] = p0[N * 3 + batch];
                p0 += batch * 2;
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[N * 2];
                pp[3] = p0[N * 3];
                p0 += batch;
                pp += 4;
            }
        }
#endif // __SSE2__
        for (; ii + 1 < max_ii; ii += 2)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[batch];
                pp[2] = p0[N];
                pp[3] = p0[N + batch];
                p0 += batch * 2;
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                p0 += batch;
                pp += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += batch;
                pp += 1;
            }
        }
    }
}

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk, int nT)
{
    #pragma omp parallel for num_threads(nT)
    for (int b = 0; b < batch; b++)
    {
        short* pp = BT.row<short>(b);

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
                __m512i _r2 = _mm512_loadu_si512((const __m512i*)(p0 + 64));
                __m512i _r3 = _mm512_loadu_si512((const __m512i*)(p0 + 96));
                __m512i _r4 = _mm512_loadu_si512((const __m512i*)(p0 + 128));
                __m512i _r5 = _mm512_loadu_si512((const __m512i*)(p0 + 160));
                __m512i _r6 = _mm512_loadu_si512((const __m512i*)(p0 + 192));
                __m512i _r7 = _mm512_loadu_si512((const __m512i*)(p0 + 224));
                __m512i _tmp0 = _mm512_shuffle_i32x4(_r0, _r2, _MM_SHUFFLE(1, 0, 1, 0));
                __m512i _tmp1 = _mm512_shuffle_i32x4(_r0, _r2, _MM_SHUFFLE(3, 2, 3, 2));
                __m512i _tmp2 = _mm512_shuffle_i32x4(_r1, _r3, _MM_SHUFFLE(1, 0, 1, 0));
                __m512i _tmp3 = _mm512_shuffle_i32x4(_r1, _r3, _MM_SHUFFLE(3, 2, 3, 2));
                __m512i _tmp4 = _mm512_shuffle_i32x4(_r4, _r6, _MM_SHUFFLE(1, 0, 1, 0));
                __m512i _tmp5 = _mm512_shuffle_i32x4(_r4, _r6, _MM_SHUFFLE(3, 2, 3, 2));
                __m512i _tmp6 = _mm512_shuffle_i32x4(_r5, _r7, _MM_SHUFFLE(1, 0, 1, 0));
                __m512i _tmp7 = _mm512_shuffle_i32x4(_r5, _r7, _MM_SHUFFLE(3, 2, 3, 2));
                _r0 = _mm512_unpacklo_epi32(_tmp0, _tmp1);
                _r1 = _mm512_unpackhi_epi32(_tmp0, _tmp1);
                _r2 = _mm512_unpacklo_epi32(_tmp2, _tmp3);
                _r3 = _mm512_unpackhi_epi32(_tmp2, _tmp3);
                _r4 = _mm512_unpacklo_epi32(_tmp4, _tmp5);
                _r5 = _mm512_unpackhi_epi32(_tmp4, _tmp5);
                _r6 = _mm512_unpacklo_epi32(_tmp6, _tmp7);
                _r7 = _mm512_unpackhi_epi32(_tmp6, _tmp7);
                _tmp0 = _mm512_unpacklo_epi64(_r0, _r2);
                _tmp1 = _mm512_unpackhi_epi64(_r0, _r2);
                _tmp2 = _mm512_unpacklo_epi64(_r1, _r3);
                _tmp3 = _mm512_unpackhi_epi64(_r1, _r3);
                _tmp4 = _mm512_unpacklo_epi64(_r4, _r6);
                _tmp5 = _mm512_unpackhi_epi64(_r4, _r6);
                _tmp6 = _mm512_unpacklo_epi64(_r5, _r7);
                _tmp7 = _mm512_unpackhi_epi64(_r5, _r7);
                _r0 = _mm512_shuffle_i32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                _r1 = _mm512_shuffle_i32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _r2 = _mm512_shuffle_i32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                _r3 = _mm512_shuffle_i32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _r4 = _mm512_shuffle_i32x4(_tmp0, _tmp4, _MM_SHUFFLE(3, 1, 3, 1));
                _r5 = _mm512_shuffle_i32x4(_tmp1, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                _r6 = _mm512_shuffle_i32x4(_tmp2, _tmp6, _MM_SHUFFLE(3, 1, 3, 1));
                _r7 = _mm512_shuffle_i32x4(_tmp3, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                _mm512_storeu_si512((__m512i*)pp, _r0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _r1);
                _mm512_storeu_si512((__m512i*)(pp + 64), _r2);
                _mm512_storeu_si512((__m512i*)(pp + 96), _r3);
                _mm512_storeu_si512((__m512i*)(pp + 128), _r4);
                _mm512_storeu_si512((__m512i*)(pp + 160), _r5);
                _mm512_storeu_si512((__m512i*)(pp + 192), _r6);
                _mm512_storeu_si512((__m512i*)(pp + 224), _r7);
                p0 += max_jj * batch * 16;
                pp += 256;
            }
            p0 -= (b * max_jj + jj) * 16;
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
                __m512i _r2 = _mm512_loadu_si512((const __m512i*)(p0 + 64));
                __m512i _r3 = _mm512_loadu_si512((const __m512i*)(p0 + 96));
                __m512i _tmp0 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(2, 0, 2, 0));
                __m512i _tmp1 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 1, 3, 1));
                __m512i _tmp2 = _mm512_shuffle_i32x4(_r2, _r3, _MM_SHUFFLE(2, 0, 2, 0));
                __m512i _tmp3 = _mm512_shuffle_i32x4(_r2, _r3, _MM_SHUFFLE(3, 1, 3, 1));
                _r0 = _mm512_unpacklo_epi32(_tmp0, _tmp1);
                _r1 = _mm512_unpackhi_epi32(_tmp0, _tmp1);
                _r2 = _mm512_unpacklo_epi32(_tmp2, _tmp3);
                _r3 = _mm512_unpackhi_epi32(_tmp2, _tmp3);
                _tmp0 = _mm512_permutex_epi64(_r0, _MM_SHUFFLE(3, 1, 2, 0));
                _tmp1 = _mm512_permutex_epi64(_r1, _MM_SHUFFLE(3, 1, 2, 0));
                _tmp2 = _mm512_permutex_epi64(_r2, _MM_SHUFFLE(3, 1, 2, 0));
                _tmp3 = _mm512_permutex_epi64(_r3, _MM_SHUFFLE(3, 1, 2, 0));
                _r0 = _mm512_shuffle_i32x4(_tmp0, _tmp2, _MM_SHUFFLE(2, 0, 2, 0));
                _r1 = _mm512_shuffle_i32x4(_tmp0, _tmp2, _MM_SHUFFLE(3, 1, 3, 1));
                _r2 = _mm512_shuffle_i32x4(_tmp1, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _r3 = _mm512_shuffle_i32x4(_tmp1, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                _mm512_storeu_si512((__m512i*)pp, _r0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _r1);
                _mm512_storeu_si512((__m512i*)(pp + 64), _r2);
                _mm512_storeu_si512((__m512i*)(pp + 96), _r3);
                p0 += max_jj * batch * 8;
                pp += 128;
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                _mm512_storeu_si512((__m512i*)pp, _r0);
                p0 += max_jj * batch * 2;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                _mm256_store_si256((__m256i*)pp, _r0);
                p0 += max_jj * batch;
                pp += 16;
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const short* p0 = B;

            int kk = 0;
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
                __m512i _r2 = _mm512_loadu_si512((const __m512i*)(p0 + 64));
                __m512i _r3 = _mm512_loadu_si512((const __m512i*)(p0 + 96));
                __m512i _tmp0 = _mm512_shuffle_i32x4(_r0, _r2, _MM_SHUFFLE(1, 0, 1, 0));
                __m512i _tmp1 = _mm512_shuffle_i32x4(_r0, _r2, _MM_SHUFFLE(3, 2, 3, 2));
                __m512i _tmp2 = _mm512_shuffle_i32x4(_r1, _r3, _MM_SHUFFLE(1, 0, 1, 0));
                __m512i _tmp3 = _mm512_shuffle_i32x4(_r1, _r3, _MM_SHUFFLE(3, 2, 3, 2));
                _r0 = _mm512_unpacklo_epi32(_tmp0, _tmp1);
                _r1 = _mm512_unpackhi_epi32(_tmp0, _tmp1);
                _r2 = _mm512_unpacklo_epi32(_tmp2, _tmp3);
                _r3 = _mm512_unpackhi_epi32(_tmp2, _tmp3);
                _tmp0 = _mm512_unpacklo_epi64(_r0, _r2);
                _tmp1 = _mm512_unpackhi_epi64(_r0, _r2);
                _tmp2 = _mm512_unpacklo_epi64(_r1, _r3);
                _tmp3 = _mm512_unpackhi_epi64(_r1, _r3);
                _r0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _r1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _r2 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                _r3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                _mm512_storeu_si512((__m512i*)pp, _r0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _r1);
                _mm512_storeu_si512((__m512i*)(pp + 64), _r2);
                _mm512_storeu_si512((__m512i*)(pp + 96), _r3);
                p0 += max_jj * batch * 16;
                pp += 128;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __AVX__
                __m256 _r0 = _mm256_loadu_ps((const float*)p0);
                __m256 _r1 = _mm256_loadu_ps((const float*)(p0 + 16));
                __m256 _r2 = _mm256_loadu_ps((const float*)(p0 + 32));
                __m256 _r3 = _mm256_loadu_ps((const float*)(p0 + 48));
                __m256 _tmp0 = _mm256_permute2f128_ps(_r0, _r2, _MM_SHUFFLE(0, 2, 0, 0));
                __m256 _tmp1 = _mm256_permute2f128_ps(_r0, _r2, _MM_SHUFFLE(0, 3, 0, 1));
                __m256 _tmp2 = _mm256_permute2f128_ps(_r1, _r3, _MM_SHUFFLE(0, 2, 0, 0));
                __m256 _tmp3 = _mm256_permute2f128_ps(_r1, _r3, _MM_SHUFFLE(0, 3, 0, 1));
                _r0 = _mm256_unpacklo_ps(_tmp0, _tmp1);
                _r1 = _mm256_unpackhi_ps(_tmp0, _tmp1);
                _r2 = _mm256_unpacklo_ps(_tmp2, _tmp3);
                _r3 = _mm256_unpackhi_ps(_tmp2, _tmp3);
                _tmp0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_r0), _mm256_castps_pd(_r2)));
                _tmp1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_r0), _mm256_castps_pd(_r2)));
                _tmp2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_r1), _mm256_castps_pd(_r3)));
                _tmp3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_r1), _mm256_castps_pd(_r3)));
                _mm256_storeu_ps((float*)pp, _tmp0);
                _mm256_storeu_ps((float*)(pp + 16), _tmp1);
                _mm256_storeu_ps((float*)(pp + 32), _tmp2);
                _mm256_storeu_ps((float*)(pp + 48), _tmp3);
#else
                __m128i _r0 = _mm_load_si128((const __m128i*)p0);
                __m128i _r1 = _mm_load_si128((const __m128i*)(p0 + 8));
                __m128i _r2 = _mm_load_si128((const __m128i*)(p0 + 8 * 2));
                __m128i _r3 = _mm_load_si128((const __m128i*)(p0 + 8 * 3));
                __m128i _r4 = _mm_load_si128((const __m128i*)(p0 + 8 * 4));
                __m128i _r5 = _mm_load_si128((const __m128i*)(p0 + 8 * 5));
                __m128i _r6 = _mm_load_si128((const __m128i*)(p0 + 8 * 6));
                __m128i _r7 = _mm_load_si128((const __m128i*)(p0 + 8 * 7));
                transpose4x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm_store_si128((__m128i*)pp, _r0);
                _mm_store_si128((__m128i*)(pp + 8), _r1);
                _mm_store_si128((__m128i*)(pp + 8 * 2), _r2);
                _mm_store_si128((__m128i*)(pp + 8 * 3), _r3);
                _mm_store_si128((__m128i*)(pp + 8 * 4), _r4);
                _mm_store_si128((__m128i*)(pp + 8 * 5), _r5);
                _mm_store_si128((__m128i*)(pp + 8 * 6), _r6);
                _mm_store_si128((__m128i*)(pp + 8 * 7), _r7);
#endif // __AVX__
                p0 += max_jj * batch * 8;
                pp += 64;
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __AVX__
                __m256 _r0 = _mm256_loadu_ps((const float*)p0);
                _mm256_storeu_ps((float*)pp, _r0);
#else
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                _mm_store_si128((__m128i*)pp, _r0);
                _mm_store_si128((__m128i*)(pp + 8), _r1);
#endif // __AVX__
                p0 += max_jj * batch * 2;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                _mm_store_si128((__m128i*)pp, _r0);
                p0 += max_jj * batch;
                pp += 8;
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const short* p0 = B;

            int kk = 0;
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
                __m512i _tmp0 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(1, 0, 1, 0));
                __m512i _tmp1 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 2, 3, 2));
                _r0 = _mm512_unpacklo_epi32(_tmp0, _tmp1);
                _r1 = _mm512_unpackhi_epi32(_tmp0, _tmp1);
                _r0 = _mm512_permutex_epi64(_r0, _MM_SHUFFLE(3, 1, 2, 0));
                _r1 = _mm512_permutex_epi64(_r1, _MM_SHUFFLE(3, 1, 2, 0));
                _tmp0 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(1, 0, 1, 0));
                _tmp1 = _mm512_shuffle_i32x4(_r0, _r1, _MM_SHUFFLE(3, 2, 3, 2));
                _r0 = _mm512_unpacklo_epi64(_tmp0, _tmp1);
                _r1 = _mm512_unpackhi_epi64(_tmp0, _tmp1);
                _mm512_storeu_si512((__m512i*)pp, _r0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _r1);
                p0 += max_jj * batch * 16;
                pp += 64;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_load_si128((const __m128i*)p0);
                __m128i _r1 = _mm_load_si128((const __m128i*)(p0 + 8));
                __m128i _r2 = _mm_load_si128((const __m128i*)(p0 + 8 * 2));
                __m128i _r3 = _mm_load_si128((const __m128i*)(p0 + 8 * 3));
                transpose4x4_epi32(_r0, _r1, _r2, _r3);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                _mm_storeu_si128((__m128i*)(pp + 8 * 2), _r2);
                _mm_storeu_si128((__m128i*)(pp + 8 * 3), _r3);
                p0 += max_jj * batch * 8;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                _mm_storeu_si128((__m128i*)pp, _r0);
                p0 += max_jj * batch * 2;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                p0 += max_jj * batch;
                pp += 4;
            }
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            const short* p0 = B;

            int kk = 0;
#if __SSE2__
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_load_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_load_si256((const __m256i*)(p0 + 16));
                transpose8x2_epi32(_r0, _r1);
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                p0 += max_jj * batch * 16;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_load_si128((const __m128i*)p0);
                __m128i _r1 = _mm_load_si128((const __m128i*)(p0 + 8));
                __m128i _tmp0 = _mm_unpacklo_epi32(_r0, _r1);
                __m128i _tmp1 = _mm_unpackhi_epi32(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _tmp0);
                _mm_storeu_si128((__m128i*)(pp + 8), _tmp1);
                p0 += max_jj * batch * 8;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __SSE2__
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                p0 += max_jj * batch * 2;
                pp += 4;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                p0 += max_jj * batch;
                pp += 2;
            }
        }
        for (; jj < max_jj; jj++)
        {
            const short* p0 = B;

            int kk = 0;
#if __SSE2__
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_load_si256((const __m256i*)p0);
                _mm256_storeu_si256((__m256i*)pp, _r0);
                p0 += max_jj * batch * 16;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_load_si128((const __m128i*)p0);
                _mm_storeu_si128((__m128i*)pp, _r0);
                p0 += max_jj * batch * 8;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __SSE2__
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                p0 += max_jj * batch * 2;
                pp += 2;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += max_jj * batch;
                pp += 1;
            }
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk, bool k_end)
{
    int* outptr = top_blob;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
            for (; jj + 15 < max_jj; jj += 16)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                    __m512i _pA1 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 1, 0, 3));
                    __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _pA3 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(0, 3, 2, 1));
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                    __m512i _pB2 = _mm512_shuffle_epi32(_pB0, _MM_PERM_BADC);
                    __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _pA0, _pB2);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _pA0, _pB3);
                    _sum4 = _mm512_dpwssd_epi32(_sum4, _pA1, _pB0);
                    _sum5 = _mm512_dpwssd_epi32(_sum5, _pA1, _pB1);
                    _sum6 = _mm512_dpwssd_epi32(_sum6, _pA1, _pB2);
                    _sum7 = _mm512_dpwssd_epi32(_sum7, _pA1, _pB3);
                    _sum8 = _mm512_dpwssd_epi32(_sum8, _pA2, _pB0);
                    _sum9 = _mm512_dpwssd_epi32(_sum9, _pA2, _pB1);
                    _suma = _mm512_dpwssd_epi32(_suma, _pA2, _pB2);
                    _sumb = _mm512_dpwssd_epi32(_sumb, _pA2, _pB3);
                    _sumc = _mm512_dpwssd_epi32(_sumc, _pA3, _pB0);
                    _sumd = _mm512_dpwssd_epi32(_sumd, _pA3, _pB1);
                    _sume = _mm512_dpwssd_epi32(_sume, _pA3, _pB2);
                    _sumf = _mm512_dpwssd_epi32(_sumf, _pA3, _pB3);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA0, _pB2));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA0, _pB3));
                    _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA1, _pB0));
                    _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA1, _pB1));
                    _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA1, _pB2));
                    _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA1, _pB3));
                    _sum8 = _mm512_add_epi32(_sum8, _mm512_madd_epi16(_pA2, _pB0));
                    _sum9 = _mm512_add_epi32(_sum9, _mm512_madd_epi16(_pA2, _pB1));
                    _suma = _mm512_add_epi32(_suma, _mm512_madd_epi16(_pA2, _pB2));
                    _sumb = _mm512_add_epi32(_sumb, _mm512_madd_epi16(_pA2, _pB3));
                    _sumc = _mm512_add_epi32(_sumc, _mm512_madd_epi16(_pA3, _pB0));
                    _sumd = _mm512_add_epi32(_sumd, _mm512_madd_epi16(_pA3, _pB1));
                    _sume = _mm512_add_epi32(_sume, _mm512_madd_epi16(_pA3, _pB2));
                    _sumf = _mm512_add_epi32(_sumf, _mm512_madd_epi16(_pA3, _pB3));
#endif

                    pA += 32;
                    pB += 32;
                }
                for (; kk < max_kk; kk++)
                {
                    __m512i _pA0 = _mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i*)pA));
                    __m512i _pB0 = _mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i*)pB));

                    __m512i _pA1 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 1, 0, 3));
                    __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _pA3 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(0, 3, 2, 1));

                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                    __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

                    __m512i _s0 = _mm512_mullo_epi32(_pA0, _pB0);
                    __m512i _s1 = _mm512_mullo_epi32(_pA0, _pB1);
                    __m512i _s2 = _mm512_mullo_epi32(_pA0, _pB2);
                    __m512i _s3 = _mm512_mullo_epi32(_pA0, _pB3);
                    __m512i _s4 = _mm512_mullo_epi32(_pA1, _pB0);
                    __m512i _s5 = _mm512_mullo_epi32(_pA1, _pB1);
                    __m512i _s6 = _mm512_mullo_epi32(_pA1, _pB2);
                    __m512i _s7 = _mm512_mullo_epi32(_pA1, _pB3);
                    __m512i _s8 = _mm512_mullo_epi32(_pA2, _pB0);
                    __m512i _s9 = _mm512_mullo_epi32(_pA2, _pB1);
                    __m512i _sa = _mm512_mullo_epi32(_pA2, _pB2);
                    __m512i _sb = _mm512_mullo_epi32(_pA2, _pB3);
                    __m512i _sc = _mm512_mullo_epi32(_pA3, _pB0);
                    __m512i _sd = _mm512_mullo_epi32(_pA3, _pB1);
                    __m512i _se = _mm512_mullo_epi32(_pA3, _pB2);
                    __m512i _sf = _mm512_mullo_epi32(_pA3, _pB3);
                    _sum0 = _mm512_add_epi32(_sum0, _s0);
                    _sum1 = _mm512_add_epi32(_sum1, _s1);
                    _sum2 = _mm512_add_epi32(_sum2, _s2);
                    _sum3 = _mm512_add_epi32(_sum3, _s3);
                    _sum4 = _mm512_add_epi32(_sum4, _s4);
                    _sum5 = _mm512_add_epi32(_sum5, _s5);
                    _sum6 = _mm512_add_epi32(_sum6, _s6);
                    _sum7 = _mm512_add_epi32(_sum7, _s7);
                    _sum8 = _mm512_add_epi32(_sum8, _s8);
                    _sum9 = _mm512_add_epi32(_sum9, _s9);
                    _suma = _mm512_add_epi32(_suma, _sa);
                    _sumb = _mm512_add_epi32(_sumb, _sb);
                    _sumc = _mm512_add_epi32(_sumc, _sc);
                    _sumd = _mm512_add_epi32(_sumd, _sd);
                    _sume = _mm512_add_epi32(_sume, _se);
                    _sumf = _mm512_add_epi32(_sumf, _sf);

                    pA += 16;
                    pB += 16;
                }

                if (k_end)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
                    //      01 12 23 30 45 56 67 74 89 9a ab b8 cd de ef fc
                    //      02 13 20 31 46 57 64 75 8a 9b a8 b9 ce df ec fd
                    //      03 10 21 32 47 54 65 76 8b 98 a9 ba cf dc ed fe
                    //      c0 d1 e2 f3 04 15 26 37 48 59 6a 7b 8c 9d ae bf
                    //      c1 d2 e3 f0 05 16 27 34 49 5a 6b 78 8d 9e af bc
                    //      c2 d3 e0 f1 06 17 24 35 4a 5b 68 79 8e 9f ac bd
                    //      c3 d0 e1 f2 07 14 25 36 4b 58 69 7a 8f 9c ad be
                    //      80 91 a2 b3 c4 d5 e6 f7 08 19 2a 3b 4c 5d 6e 7f
                    //      81 92 a3 b0 c5 d6 e7 f4 09 1a 2b 38 4d 5e 6f 7c
                    //      82 93 a0 b1 c6 d7 e4 f5 0a 1b 28 39 4e 5f 6c 7d
                    //      83 90 a1 b2 c7 d4 e5 f6 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 84 95 a6 b7 c8 d9 ea fb 0c 1d 2e 3f
                    //      41 52 63 70 85 96 a7 b4 c9 da eb f8 0d 1e 2f 3c
                    //      42 53 60 71 86 97 a4 b5 ca db e8 f9 0e 1f 2c 3d
                    //      43 50 61 72 87 94 a5 b6 cb d8 e9 fa 0f 1c 2d 3e
                    // to
                    //      00 10 20 30 44 54 64 74 88 98 a8 b8 cc dc ec fc
                    //      01 11 21 31 45 55 65 75 89 99 a9 b9 cd dd ed fd
                    //      02 12 22 32 46 56 66 76 8a 9a aa ba ce de ee fe
                    //      03 13 23 33 47 57 67 77 8b 9b ab bb cf df ef ff
                    //      c0 d0 e0 f0 04 14 24 34 48 58 68 78 8c 9c ac bc
                    //      c1 d1 e1 f1 05 15 25 35 49 59 69 79 8d 9d ad bd
                    //      c2 d2 e2 f2 06 16 26 36 4a 5a 6a 7a 8e 9e ae be
                    //      c3 d3 e3 f3 07 17 27 37 4b 5b 6b 7b 8f 9f af bf
                    //      80 90 a0 b0 c4 d4 e4 f4 08 18 28 38 4c 5c 6c 7c
                    //      81 91 a1 b1 c5 d5 e5 f5 09 19 29 39 4d 5d 6d 7d
                    //      82 92 a2 b2 c6 d6 e6 f6 0a 1a 2a 3a 4e 5e 6e 7e
                    //      83 93 a3 b3 c7 d7 e7 f7 0b 1b 2b 3b 4f 5f 6f 7f
                    //      40 50 60 70 84 94 a4 b4 c8 d8 e8 f8 0c 1c 2c 3c
                    //      41 51 61 71 85 95 a5 b5 c9 d9 e9 f9 0d 1d 2d 3d
                    //      42 52 62 72 86 96 a6 b6 ca da ea fa 0e 1e 2e 3e
                    //      43 53 63 73 87 97 a7 b7 cb db eb fb 0f 1f 2f 3f
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _suma = _mm512_shuffle_epi32(_suma, _MM_PERM_BADC);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_ADCB);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sume = _mm512_shuffle_epi32(_sume, _MM_PERM_BADC);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        __m512i _tmp8 = _mm512_unpacklo_epi32(_sum8, _sumb);
                        __m512i _tmp9 = _mm512_unpackhi_epi32(_sum8, _sumb);
                        __m512i _tmpa = _mm512_unpacklo_epi32(_suma, _sum9);
                        __m512i _tmpb = _mm512_unpackhi_epi32(_suma, _sum9);
                        __m512i _tmpc = _mm512_unpacklo_epi32(_sumc, _sumf);
                        __m512i _tmpd = _mm512_unpackhi_epi32(_sumc, _sumf);
                        __m512i _tmpe = _mm512_unpacklo_epi32(_sume, _sumd);
                        __m512i _tmpf = _mm512_unpackhi_epi32(_sume, _sumd);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum8 = _mm512_unpacklo_epi64(_tmp8, _tmpa);
                        _sum9 = _mm512_unpackhi_epi64(_tmp8, _tmpa);
                        _suma = _mm512_unpacklo_epi64(_tmpb, _tmp9);
                        _sumb = _mm512_unpackhi_epi64(_tmpb, _tmp9);
                        _sumc = _mm512_unpacklo_epi64(_tmpc, _tmpe);
                        _sumd = _mm512_unpackhi_epi64(_tmpc, _tmpe);
                        _sume = _mm512_unpacklo_epi64(_tmpf, _tmpd);
                        _sumf = _mm512_unpackhi_epi64(_tmpf, _tmpd);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_CBAD);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sumc, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum4, _sum0, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum8, _sum4, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sumc, _sum8, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum1, _sumd, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum5, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum9, _sum5, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sumd, _sum9, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp8 = _mm512_shuffle_i32x4(_sum2, _sume, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp9 = _mm512_shuffle_i32x4(_sum6, _sum2, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpa = _mm512_shuffle_i32x4(_suma, _sum6, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmpb = _mm512_shuffle_i32x4(_sume, _suma, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmpc = _mm512_shuffle_i32x4(_sum3, _sumf, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmpd = _mm512_shuffle_i32x4(_sum7, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpe = _mm512_shuffle_i32x4(_sumb, _sum7, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmpf = _mm512_shuffle_i32x4(_sumf, _sumb, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp4, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp8, _tmpa, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmpc, _tmpe, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp1, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp9, _tmpb, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmpd, _tmpf, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum8 = _mm512_shuffle_i32x4(_tmp2, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum9 = _mm512_shuffle_i32x4(_tmp6, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _suma = _mm512_shuffle_i32x4(_tmpa, _tmp8, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumb = _mm512_shuffle_i32x4(_tmpe, _tmpc, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumc = _mm512_shuffle_i32x4(_tmp3, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumd = _mm512_shuffle_i32x4(_tmp7, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _sume = _mm512_shuffle_i32x4(_tmpb, _tmp9, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumf = _mm512_shuffle_i32x4(_tmpf, _tmpd, _MM_SHUFFLE(3, 1, 2, 0));
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
                const short* pA = pAT;

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
                    _sum2 = _mm512_load_si512((const __m512i*)(outptr + 16 * 2));
                    _sum3 = _mm512_load_si512((const __m512i*)(outptr + 16 * 3));
                    _sum4 = _mm512_load_si512((const __m512i*)(outptr + 16 * 4));
                    _sum5 = _mm512_load_si512((const __m512i*)(outptr + 16 * 5));
                    _sum6 = _mm512_load_si512((const __m512i*)(outptr + 16 * 6));
                    _sum7 = _mm512_load_si512((const __m512i*)(outptr + 16 * 7));
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                    __m512i _pA1 = _mm512_permutex_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _pB0 = _mm512_inserti32x8(_mm512_castsi256_si512(_pB), _pB, 1);
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                    __m512i _pB2 = _mm512_shuffle_epi32(_pB0, _MM_PERM_BADC);
                    __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _pA0, _pB2);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _pA0, _pB3);
                    _sum4 = _mm512_dpwssd_epi32(_sum4, _pA1, _pB0);
                    _sum5 = _mm512_dpwssd_epi32(_sum5, _pA1, _pB1);
                    _sum6 = _mm512_dpwssd_epi32(_sum6, _pA1, _pB2);
                    _sum7 = _mm512_dpwssd_epi32(_sum7, _pA1, _pB3);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA0, _pB2));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA0, _pB3));
                    _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA1, _pB0));
                    _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA1, _pB1));
                    _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA1, _pB2));
                    _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA1, _pB3));
#endif

                    pA += 32;
                    pB += 16;
                }
                for (; kk < max_kk; kk++)
                {
                    __m512i _pA0 = _mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i*)pA));
                    __m256i _pB = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)pB));
                    __m512i _pA1 = _mm512_permutex_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _pB0 = _mm512_inserti32x8(_mm512_castsi256_si512(_pB), _pB, 1);
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                    __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

                    __m512i _s0 = _mm512_mullo_epi32(_pA0, _pB0);
                    __m512i _s1 = _mm512_mullo_epi32(_pA0, _pB1);
                    __m512i _s2 = _mm512_mullo_epi32(_pA0, _pB2);
                    __m512i _s3 = _mm512_mullo_epi32(_pA0, _pB3);
                    __m512i _s4 = _mm512_mullo_epi32(_pA1, _pB0);
                    __m512i _s5 = _mm512_mullo_epi32(_pA1, _pB1);
                    __m512i _s6 = _mm512_mullo_epi32(_pA1, _pB2);
                    __m512i _s7 = _mm512_mullo_epi32(_pA1, _pB3);
                    _sum0 = _mm512_add_epi32(_sum0, _s0);
                    _sum1 = _mm512_add_epi32(_sum1, _s1);
                    _sum2 = _mm512_add_epi32(_sum2, _s2);
                    _sum3 = _mm512_add_epi32(_sum3, _s3);
                    _sum4 = _mm512_add_epi32(_sum4, _s4);
                    _sum5 = _mm512_add_epi32(_sum5, _s5);
                    _sum6 = _mm512_add_epi32(_sum6, _s6);
                    _sum7 = _mm512_add_epi32(_sum7, _s7);

                    pA += 16;
                    pB += 8;
                }

                if (k_end)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 80 91 a2 b3 c4 d5 e6 f7
                    //      01 12 23 30 45 56 67 74 81 92 a3 b0 c5 d6 e7 f4
                    //      02 13 20 31 46 57 64 75 82 93 a0 b1 c6 d7 e4 f5
                    //      03 10 21 32 47 54 65 76 83 90 a1 b2 c7 d4 e5 f6
                    //      40 51 62 73 04 15 26 37 c0 d1 e2 f3 84 95 a6 b7
                    //      41 52 63 70 05 16 27 34 c1 d2 e3 f0 85 96 a7 b4
                    //      42 53 60 71 06 17 24 35 c2 d3 e0 f1 86 97 a4 b5
                    //      43 50 61 72 07 14 25 36 c3 d0 e1 f2 87 94 a5 b6
                    // to
                    //      00 10 20 30 44 54 64 74 80 90 a0 b0 c4 d4 e4 f4
                    //      01 11 21 31 45 55 65 75 81 91 a1 b1 c5 d5 e5 f5
                    //      02 12 22 32 46 56 66 76 82 92 a2 b2 c6 d6 e6 f6
                    //      03 13 23 33 47 57 67 77 83 93 a3 b3 c7 d7 e7 f7
                    //      40 50 60 70 04 14 24 34 c0 d0 e0 f0 84 94 a4 b4
                    //      41 51 61 71 05 15 25 35 c1 d1 e1 f1 85 95 a5 b5
                    //      42 52 62 72 06 16 26 36 c2 d2 e2 f2 86 96 a6 b6
                    //      43 53 63 73 07 17 27 37 c3 d3 e3 f3 87 97 a7 b7
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    // TODO
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum4, _sum0, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum5, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum6, _sum2, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum7, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));
                }

                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
                _mm512_store_si512((__m512i*)(outptr + 16 * 2), _sum2);
                _mm512_store_si512((__m512i*)(outptr + 16 * 3), _sum3);
                _mm512_store_si512((__m512i*)(outptr + 16 * 4), _sum4);
                _mm512_store_si512((__m512i*)(outptr + 16 * 5), _sum5);
                _mm512_store_si512((__m512i*)(outptr + 16 * 6), _sum6);
                _mm512_store_si512((__m512i*)(outptr + 16 * 7), _sum7);
                outptr += 16 * 8;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

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
                    _sum2 = _mm512_load_si512((const __m512i*)(outptr + 16 * 2));
                    _sum3 = _mm512_load_si512((const __m512i*)(outptr + 16 * 3));
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)pB));
                    __m512i _pB0 = _mm512_shuffle_i32x4(_pB, _pB, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _pA1, _pB1);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA1, _pB0));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA1, _pB1));
#endif

                    pA += 32;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m512i _pA0 = _mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i*)pA));
                    __m512i _pB0 = _mm512_cvtepi16_epi32(_mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB)));
                    __m512i _pA1 = _mm512_permutex_epi64(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                    __m512i _s0 = _mm512_mullo_epi32(_pA0, _pB0);
                    __m512i _s1 = _mm512_mullo_epi32(_pA0, _pB1);
                    __m512i _s2 = _mm512_mullo_epi32(_pA1, _pB0);
                    __m512i _s3 = _mm512_mullo_epi32(_pA1, _pB1);
                    _sum0 = _mm512_add_epi32(_sum0, _s0);
                    _sum1 = _mm512_add_epi32(_sum1, _s1);
                    _sum2 = _mm512_add_epi32(_sum2, _s2);
                    _sum3 = _mm512_add_epi32(_sum3, _s3);

                    pA += 16;
                    pB += 4;
                }

                if (k_end)
                {
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
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }
                }

                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
                _mm512_store_si512((__m512i*)(outptr + 16 * 2), _sum2);
                _mm512_store_si512((__m512i*)(outptr + 16 * 3), _sum3);
                outptr += 16 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB0 = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pB)[0]));
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CDAB);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA, _pB0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _pA, _pB1);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA, _pB0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA, _pB1));
#endif

                    pA += 32;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    __m512i _pA = _mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i*)pA));
                    __m512i _pB0 = _mm512_cvtepi16_epi32(_mm256_set1_epi32(((const int*)pB)[0]));
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ABAB);

                    __m512i _s0 = _mm512_mullo_epi32(_pA, _pB0);
                    __m512i _s1 = _mm512_mullo_epi32(_pA, _pB1);
                    _sum0 = _mm512_add_epi32(_sum0, _s0);
                    _sum1 = _mm512_add_epi32(_sum1, _s1);

                    pA += 16;
                    pB += 2;
                }

                if (k_end)
                {
                    // from
                    //      00 11 20 31 40 51 60 71 80 91 a0 b1 c0 d1 e0 f1
                    //      01 10 21 30 41 50 61 70 81 90 a1 b0 c1 d0 e1 f0
                    // to
                    //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                    //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                    {
                        __m512i _tmp0 = _mm512_shuffle_epi32(_sum0, _MM_PERM_DBCA);
                        __m512i _tmp1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_ACDB);
                        _sum0 = _mm512_unpacklo_epi32(_tmp0, _tmp1);
                        _sum1 = _mm512_unpackhi_epi32(_tmp0, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                    }
                }

                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
                outptr += 16 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                    __m512i _pB = _mm512_set1_epi32(((const int*)pB)[0]);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA, _pB);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA, _pB));
#endif

                    pA += 32;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    __m512i _pA = _mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i*)pA));
                    __m512i _pB = _mm512_set1_epi32(pB[0]);
                    __m512i _s0 = _mm512_mullo_epi32(_pA, _pB);
                    _sum0 = _mm512_add_epi32(_sum0, _s0);

                    pA += 16;
                    pB += 1;
                }

                _mm512_store_si512((__m512i*)outptr, _sum0);
                outptr += 16;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
            for (; jj + 15 < max_jj; jj += 16)
            {
                const short* pA = pAT;

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
                    _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
                    _sum1 = _mm512_loadu_si512((const __m512i*)(outptr + 16));
                    _sum2 = _mm512_loadu_si512((const __m512i*)(outptr + 32));
                    _sum3 = _mm512_loadu_si512((const __m512i*)(outptr + 48));
                    _sum4 = _mm512_loadu_si512((const __m512i*)(outptr + 64));
                    _sum5 = _mm512_loadu_si512((const __m512i*)(outptr + 80));
                    _sum6 = _mm512_loadu_si512((const __m512i*)(outptr + 96));
                    _sum7 = _mm512_loadu_si512((const __m512i*)(outptr + 112));
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);

                    __m512i _pA00 = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);
                    __m512i _pA11 = _mm512_permutex_epi64(_pA00, _MM_SHUFFLE(1, 0, 3, 2));
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                    __m512i _pB2 = _mm512_shuffle_epi32(_pB0, _MM_PERM_BADC);
                    __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA00, _pB0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _pA00, _pB1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _pA00, _pB2);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _pA00, _pB3);
                    _sum4 = _mm512_dpwssd_epi32(_sum4, _pA11, _pB0);
                    _sum5 = _mm512_dpwssd_epi32(_sum5, _pA11, _pB1);
                    _sum6 = _mm512_dpwssd_epi32(_sum6, _pA11, _pB2);
                    _sum7 = _mm512_dpwssd_epi32(_sum7, _pA11, _pB3);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA00, _pB0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA00, _pB1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA00, _pB2));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA00, _pB3));
                    _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA11, _pB0));
                    _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA11, _pB1));
                    _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA11, _pB2));
                    _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA11, _pB3));
#endif // __AVX512VNNI__

                    pA += 16;
                    pB += 32;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = _mm_load_si128((const __m128i*)pA);
                    __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                    __m256i _pA0 = _mm256_cvtepi16_epi32(_pA);
                    __m512i _pB0 = _mm512_cvtepi16_epi32(_pB);

                    __m512i _pA00 = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);
                    __m512i _pA11 = _mm512_permutex_epi64(_pA00, _MM_SHUFFLE(1, 0, 3, 2));

                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                    __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

                    __m512i _s0 = _mm512_mullo_epi32(_pA00, _pB0);
                    __m512i _s1 = _mm512_mullo_epi32(_pA00, _pB1);
                    __m512i _s2 = _mm512_mullo_epi32(_pA00, _pB2);
                    __m512i _s3 = _mm512_mullo_epi32(_pA00, _pB3);
                    __m512i _s4 = _mm512_mullo_epi32(_pA11, _pB0);
                    __m512i _s5 = _mm512_mullo_epi32(_pA11, _pB1);
                    __m512i _s6 = _mm512_mullo_epi32(_pA11, _pB2);
                    __m512i _s7 = _mm512_mullo_epi32(_pA11, _pB3);
                    _sum0 = _mm512_add_epi32(_sum0, _s0);
                    _sum1 = _mm512_add_epi32(_sum1, _s1);
                    _sum2 = _mm512_add_epi32(_sum2, _s2);
                    _sum3 = _mm512_add_epi32(_sum3, _s3);
                    _sum4 = _mm512_add_epi32(_sum4, _s4);
                    _sum5 = _mm512_add_epi32(_sum5, _s5);
                    _sum6 = _mm512_add_epi32(_sum6, _s6);
                    _sum7 = _mm512_add_epi32(_sum7, _s7);

                    pA += 8;
                    pB += 16;
                }

                if (k_end)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 08 19 2a 3b 4c 5d 6e 7f
                    //      01 12 23 30 45 56 67 74 09 1a 2b 38 4d 5e 6f 7c
                    //      02 13 20 31 46 57 64 75 0a 1b 28 39 4e 5f 6c 7d
                    //      03 10 21 32 47 54 65 76 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 04 15 26 37 48 59 6a 7b 0c 1d 2e 3f
                    //      41 52 63 70 05 16 27 34 49 5a 6b 78 0d 1e 2f 3c
                    //      42 53 60 71 06 17 24 35 4a 5b 68 79 0e 1f 2c 3d
                    //      43 50 61 72 07 14 25 36 4b 58 69 7a 0f 1c 2d 3e
                    // to
                    //      00 10 20 30 44 54 64 74 08 18 28 38 4c 5c 6c 7c
                    //      01 11 21 31 45 55 65 75 09 19 29 39 4d 5d 6d 7d
                    //      02 12 22 32 46 56 66 76 0a 1a 2a 3a 4e 5e 6e 7e
                    //      03 13 23 33 47 57 67 77 0b 1b 2b 3b 4f 5f 6f 7f
                    //      40 50 60 70 04 14 24 34 48 58 68 78 0c 1c 2c 3c
                    //      41 51 61 71 05 15 25 35 49 59 69 79 0d 1d 2d 3d
                    //      42 52 62 72 06 16 26 36 4a 5a 6a 7a 0e 1e 2e 3e
                    //      43 53 63 73 07 17 27 37 4b 5b 6b 7b 0f 1f 2f 3f
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp2, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp4, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp0, _tmp2, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum3 = _mm512_shuffle_i32x4(_tmp4, _tmp6, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum4 = _mm512_shuffle_i32x4(_tmp1, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp1, _tmp3, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum7 = _mm512_shuffle_i32x4(_tmp5, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
                }

                _mm512_storeu_si512((__m512i*)outptr, _sum0);
                _mm512_storeu_si512((__m512i*)(outptr + 16), _sum1);
                _mm512_storeu_si512((__m512i*)(outptr + 32), _sum2);
                _mm512_storeu_si512((__m512i*)(outptr + 48), _sum3);
                _mm512_storeu_si512((__m512i*)(outptr + 64), _sum4);
                _mm512_storeu_si512((__m512i*)(outptr + 80), _sum5);
                _mm512_storeu_si512((__m512i*)(outptr + 96), _sum6);
                _mm512_storeu_si512((__m512i*)(outptr + 112), _sum7);
                outptr += 128;
            }
#endif // __AVX512F__
            for (; jj + 7 < max_jj; jj += 8)
            {
                const short* pA = pAT;

#if __AVX512F__
                __m512i _sum0;
                __m512i _sum1;
                __m512i _sum2;
                __m512i _sum3;
#else
                __m256i _sum0;
                __m256i _sum1;
                __m256i _sum2;
                __m256i _sum3;
                __m256i _sum4;
                __m256i _sum5;
                __m256i _sum6;
                __m256i _sum7;
#endif // __AVX512F__

                if (k == 0)
                {
#if __AVX512F__
                    _sum0 = _mm512_setzero_si512();
                    _sum1 = _mm512_setzero_si512();
                    _sum2 = _mm512_setzero_si512();
                    _sum3 = _mm512_setzero_si512();
#else
                    _sum0 = _mm256_setzero_si256();
                    _sum1 = _mm256_setzero_si256();
                    _sum2 = _mm256_setzero_si256();
                    _sum3 = _mm256_setzero_si256();
                    _sum4 = _mm256_setzero_si256();
                    _sum5 = _mm256_setzero_si256();
                    _sum6 = _mm256_setzero_si256();
                    _sum7 = _mm256_setzero_si256();
#endif // __AVX512F__
                }
                else
                {
#if __AVX512F__
                    _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
                    _sum1 = _mm512_loadu_si512((const __m512i*)(outptr + 16));
                    _sum2 = _mm512_loadu_si512((const __m512i*)(outptr + 32));
                    _sum3 = _mm512_loadu_si512((const __m512i*)(outptr + 48));
#else
                    _sum0 = _mm256_load_si256((const __m256i*)outptr);
                    _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                    _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                    _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
                    _sum4 = _mm256_load_si256((const __m256i*)(outptr + 32));
                    _sum5 = _mm256_load_si256((const __m256i*)(outptr + 40));
                    _sum6 = _mm256_load_si256((const __m256i*)(outptr + 48));
                    _sum7 = _mm256_load_si256((const __m256i*)(outptr + 56));
#endif // __AVX512F__
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pB);
#if __AVX512F__
                    __m512i _pA00 = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);
                    __m512i _pA11 = _mm512_permutex_epi64(_pA00, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    __m512i _pB01 = _mm512_inserti32x8(_mm512_castsi256_si512(_pB0), _pB1, 1);
                    __m512i _pB23 = _mm512_shuffle_epi32(_pB01, _MM_PERM_BADC);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA00, _pB01);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _pA00, _pB23);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _pA11, _pB01);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _pA11, _pB23);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA00, _pB01));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA00, _pB23));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA11, _pB01));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA11, _pB23));
#endif // __AVX512VNNI__
#else  // __AVX512F__
                    __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB3 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 1, 0, 3));

#if __AVXVNNI__
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _pA0, _pB2);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _pA0, _pB3);
                    _sum4 = _mm256_dpwssd_epi32(_sum4, _pA1, _pB0);
                    _sum5 = _mm256_dpwssd_epi32(_sum5, _pA1, _pB1);
                    _sum6 = _mm256_dpwssd_epi32(_sum6, _pA1, _pB2);
                    _sum7 = _mm256_dpwssd_epi32(_sum7, _pA1, _pB3);
#else
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA0, _pB2));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA0, _pB3));
                    _sum4 = _mm256_add_epi32(_sum4, _mm256_madd_epi16(_pA1, _pB0));
                    _sum5 = _mm256_add_epi32(_sum5, _mm256_madd_epi16(_pA1, _pB1));
                    _sum6 = _mm256_add_epi32(_sum6, _mm256_madd_epi16(_pA1, _pB2));
                    _sum7 = _mm256_add_epi32(_sum7, _mm256_madd_epi16(_pA1, _pB3));
#endif // __AVXVNNI__
#endif // __AVX512F__

                    pA += 16;
                    pB += 16;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = _mm_load_si128((const __m128i*)pA);
                    __m128i _pB = _mm_load_si128((const __m128i*)pB);

                    __m256i _pA0 = _mm256_cvtepi16_epi32(_pA);
                    __m256i _pB0 = _mm256_cvtepi16_epi32(_pB);
#if __AVX512F__
                    __m512i _pA00 = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);
                    __m512i _pA11 = _mm512_shuffle_i32x4(_pA00, _pA00, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    __m512i _pB01 = _mm512_inserti32x8(_mm512_castsi256_si512(_pB0), _pB1, 1);
                    __m512i _pB23 = _mm512_permutex_epi64(_pB01, _MM_SHUFFLE(2, 3, 0, 1));

                    __m512i _s01 = _mm512_mullo_epi32(_pA00, _pB01);
                    __m512i _s23 = _mm512_mullo_epi32(_pA00, _pB23);
                    __m512i _s45 = _mm512_mullo_epi32(_pA11, _pB01);
                    __m512i _s67 = _mm512_mullo_epi32(_pA11, _pB23);
                    _sum0 = _mm512_add_epi32(_sum0, _s01);
                    _sum1 = _mm512_add_epi32(_sum1, _s23);
                    _sum2 = _mm512_add_epi32(_sum2, _s45);
                    _sum3 = _mm512_add_epi32(_sum3, _s67);
#else
                    __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB3 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 1, 0, 3));

                    __m256i _s0 = _mm256_mullo_epi32(_pA0, _pB0);
                    __m256i _s1 = _mm256_mullo_epi32(_pA0, _pB1);
                    __m256i _s2 = _mm256_mullo_epi32(_pA0, _pB2);
                    __m256i _s3 = _mm256_mullo_epi32(_pA0, _pB3);
                    __m256i _s4 = _mm256_mullo_epi32(_pA1, _pB0);
                    __m256i _s5 = _mm256_mullo_epi32(_pA1, _pB1);
                    __m256i _s6 = _mm256_mullo_epi32(_pA1, _pB2);
                    __m256i _s7 = _mm256_mullo_epi32(_pA1, _pB3);
                    _sum0 = _mm256_add_epi32(_sum0, _s0);
                    _sum1 = _mm256_add_epi32(_sum1, _s1);
                    _sum2 = _mm256_add_epi32(_sum2, _s2);
                    _sum3 = _mm256_add_epi32(_sum3, _s3);
                    _sum4 = _mm256_add_epi32(_sum4, _s4);
                    _sum5 = _mm256_add_epi32(_sum5, _s5);
                    _sum6 = _mm256_add_epi32(_sum6, _s6);
                    _sum7 = _mm256_add_epi32(_sum7, _s7);
#endif // __AVX512F__

                    pA += 8;
                    pB += 8;
                }

#if __AVX512F__
                if (k_end)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 01 12 23 30 45 56 67 74
                    //      02 13 20 31 46 57 64 75 03 10 21 32 47 54 65 76
                    //      40 51 62 73 04 15 26 37 41 52 63 70 05 16 27 34
                    //      42 53 60 71 06 17 24 35 43 50 61 72 07 14 25 36
                    // to
                    //      00 10 20 30 44 54 64 74 04 14 24 34 40 50 60 70
                    //      01 11 21 31 45 55 65 75 05 15 25 35 41 51 61 71
                    //      02 12 22 32 46 56 66 76 06 16 26 36 42 52 62 72
                    //      03 13 23 33 47 57 67 77 07 17 27 37 43 53 63 73
                    {
                        __m512i _s0 = _mm512_shuffle_i32x4(_sum0, _sum2, _MM_SHUFFLE(0, 1, 1, 0));
                        __m512i _s1 = _mm512_shuffle_i32x4(_sum1, _sum3, _MM_SHUFFLE(2, 3, 3, 2));
                        __m512i _s2 = _mm512_shuffle_i32x4(_sum1, _sum3, _MM_SHUFFLE(0, 1, 1, 0));
                        __m512i _s3 = _mm512_shuffle_i32x4(_sum0, _sum2, _MM_SHUFFLE(2, 3, 3, 2));
                        _s1 = _mm512_shuffle_epi32(_s1, _MM_PERM_ADCB);
                        _s2 = _mm512_shuffle_epi32(_s2, _MM_PERM_BADC);
                        _s3 = _mm512_shuffle_epi32(_s3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_s0, _s1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_s0, _s1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_s2, _s3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_s2, _s3);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 0, 3, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 0, 3, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 2, 1, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(1, 2, 1, 2));
                    _sum0 = _tmp0;
                    _sum1 = _tmp1;
                    _sum2 = _tmp2;
                    _sum3 = _tmp3;
                }

                _mm512_storeu_si512((__m512i*)outptr, _sum0);
                _mm512_storeu_si512((__m512i*)(outptr + 16), _sum1);
                _mm512_storeu_si512((__m512i*)(outptr + 32), _sum2);
                _mm512_storeu_si512((__m512i*)(outptr + 48), _sum3);
                outptr += 64;
#else
                if (k_end)
                {
                    // from
                    //      00 11 22 33 44 55 66 77
                    //      01 12 23 30 45 56 67 74
                    //      02 13 20 31 46 57 64 75
                    //      03 10 21 32 47 54 65 76
                    //      40 51 62 73 04 15 26 37
                    //      41 52 63 70 05 16 27 34
                    //      42 53 60 71 06 17 24 35
                    //      43 50 61 72 07 14 25 36
                    // to
                    //      00 10 20 30 44 54 64 74
                    //      01 11 21 31 45 55 65 75
                    //      02 12 22 32 46 56 66 76
                    //      03 13 23 33 47 57 67 77
                    //      40 50 60 70 04 14 24 34
                    //      41 51 61 71 05 15 25 35
                    //      42 52 62 72 06 16 26 36
                    //      43 53 63 73 07 17 27 37
                    {
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum2 = _mm256_shuffle_epi32(_sum2, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum5 = _mm256_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum6 = _mm256_shuffle_epi32(_sum6, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum7 = _mm256_shuffle_epi32(_sum7, _MM_SHUFFLE(0, 3, 2, 1));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum3);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum3);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum1);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum1);
                        __m256i _tmp4 = _mm256_unpacklo_epi32(_sum4, _sum7);
                        __m256i _tmp5 = _mm256_unpackhi_epi32(_sum4, _sum7);
                        __m256i _tmp6 = _mm256_unpacklo_epi32(_sum6, _sum5);
                        __m256i _tmp7 = _mm256_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm256_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm256_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm256_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm256_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum5 = _mm256_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm256_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    __m256i _tmp0 = _mm256_permute2x128_si256(_sum0, _sum4, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2x128_si256(_sum1, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp2 = _mm256_permute2x128_si256(_sum2, _sum6, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp3 = _mm256_permute2x128_si256(_sum3, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp4 = _mm256_permute2x128_si256(_sum4, _sum0, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp5 = _mm256_permute2x128_si256(_sum5, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp6 = _mm256_permute2x128_si256(_sum6, _sum2, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp7 = _mm256_permute2x128_si256(_sum7, _sum3, _MM_SHUFFLE(0, 3, 0, 1));
                    _sum0 = _tmp0;
                    _sum1 = _tmp1;
                    _sum2 = _tmp2;
                    _sum3 = _tmp3;
                    _sum4 = _tmp4;
                    _sum5 = _tmp5;
                    _sum6 = _tmp6;
                    _sum7 = _tmp7;
                }

                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
                _mm256_store_si256((__m256i*)(outptr + 8 * 2), _sum2);
                _mm256_store_si256((__m256i*)(outptr + 8 * 3), _sum3);
                _mm256_store_si256((__m256i*)(outptr + 8 * 4), _sum4);
                _mm256_store_si256((__m256i*)(outptr + 8 * 5), _sum5);
                _mm256_store_si256((__m256i*)(outptr + 8 * 6), _sum6);
                _mm256_store_si256((__m256i*)(outptr + 8 * 7), _sum7);
                outptr += 8 * 8;
#endif // __AVX512F__
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                    __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pB), _pB, 1);
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

#if __AVXVNNI__ || __AVX512VNNI__
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _pA1, _pB1);
#else
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA1, _pB0));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA1, _pB1));
#endif

                    pA += 16;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m256i _pA0 = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)pA));
                    __m256i _pB0 = _mm256_cvtepi16_epi32(_mm_castpd_si128(_mm_load1_pd((const double*)pB)));
                    __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                    __m256i _s0 = _mm256_mullo_epi32(_pA0, _pB0);
                    __m256i _s1 = _mm256_mullo_epi32(_pA0, _pB1);
                    __m256i _s2 = _mm256_mullo_epi32(_pA1, _pB0);
                    __m256i _s3 = _mm256_mullo_epi32(_pA1, _pB1);
                    _sum0 = _mm256_add_epi32(_sum0, _s0);
                    _sum1 = _mm256_add_epi32(_sum1, _s1);
                    _sum2 = _mm256_add_epi32(_sum2, _s2);
                    _sum3 = _mm256_add_epi32(_sum3, _s3);

                    pA += 8;
                    pB += 4;
                }

                if (k_end)
                {
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
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum3);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum3);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum1);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }
                }

                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
                _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
                _mm256_store_si256((__m256i*)(outptr + 24), _sum3);
                outptr += 32;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pB0 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 1, 0, 1));

#if __AVXVNNI__ || __AVX512VNNI__
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _pA, _pB0);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _pA, _pB1);
#else
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA, _pB0));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA, _pB1));
#endif

                    pA += 16;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    __m256i _pA = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)pA));
                    __m256i _pB0 = _mm256_cvtepi16_epi32(_mm_castps_si128(_mm_load1_ps((const float*)pB)));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 1, 0, 1));

                    __m256i _s0 = _mm256_mullo_epi32(_pA, _pB0);
                    __m256i _s1 = _mm256_mullo_epi32(_pA, _pB1);
                    _sum0 = _mm256_add_epi32(_sum0, _s0);
                    _sum1 = _mm256_add_epi32(_sum1, _s1);

                    pA += 8;
                    pB += 2;
                }

                if (k_end)
                {
                    // from
                    //      00 11 20 31 40 51 60 71
                    //      01 10 21 30 41 50 61 70
                    // to
                    //      00 10 20 30 40 50 60 70
                    //      01 11 21 31 41 51 61 71
                    {
                        __m256i _tmp0 = _mm256_shuffle_epi32(_sum0, _MM_SHUFFLE(3, 1, 2, 0));
                        __m256i _tmp1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(0, 2, 3, 1));
                        _sum0 = _mm256_unpacklo_epi32(_tmp0, _tmp1);
                        _sum1 = _mm256_unpackhi_epi32(_tmp0, _tmp1);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                    }
                }

                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
                outptr += 16;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                    __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA, _pB));

                    pA += 16;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    __m256i _pA = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)pA));
                    __m256i _pB = _mm256_set1_epi32(pB[0]);
                    __m256i _s0 = _mm256_mullo_epi32(_pA, _pB);
                    _sum0 = _mm256_add_epi32(_sum0, _s0);

                    pA += 8;
                    pB += 1;
                }

                _mm256_store_si256((__m256i*)outptr, _sum0);
                outptr += 8;
            }
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
            for (; jj + 15 < max_jj; jj += 16)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                    __m256i _pAA = _mm256_inserti128_si256(_mm256_castsi128_si256(_pA), _pA, 1);
                    __m512i _pA0 = _mm512_inserti32x8(_mm512_castsi256_si512(_pAA), _pAA, 1);
                    __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _pA1, _pB1);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA1, _pB0));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA1, _pB1));
#endif

                    pA += 8;
                    pB += 32;
                }
                for (; kk < max_kk; kk++)
                {
                    __m256i _pA = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pA));
                    __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                    __m512i _pA0 = _mm512_cvtepi16_epi32(_pA);
                    __m512i _pB0 = _mm512_cvtepi16_epi32(_pB);
                    __m512i _pA1 = _mm512_permutex_epi64(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                    __m512i _s0 = _mm512_mullo_epi32(_pA0, _pB0);
                    __m512i _s1 = _mm512_mullo_epi32(_pA0, _pB1);
                    __m512i _s2 = _mm512_mullo_epi32(_pA1, _pB0);
                    __m512i _s3 = _mm512_mullo_epi32(_pA1, _pB1);

                    _sum0 = _mm512_add_epi32(_sum0, _s0);
                    _sum1 = _mm512_add_epi32(_sum1, _s1);
                    _sum2 = _mm512_add_epi32(_sum2, _s2);
                    _sum3 = _mm512_add_epi32(_sum3, _s3);

                    pA += 4;
                    pB += 16;
                }

                if (k_end)
                {
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
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
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
                const short* pA = pAT;

#if __AVX2__
                __m256i _sum0;
                __m256i _sum1;
                __m256i _sum2;
                __m256i _sum3;
#else
                __m128i _sum0;
                __m128i _sum1;
                __m128i _sum2;
                __m128i _sum3;
                __m128i _sum4;
                __m128i _sum5;
                __m128i _sum6;
                __m128i _sum7;
#endif

                if (k == 0)
                {
#if __AVX2__
                    _sum0 = _mm256_setzero_si256();
                    _sum1 = _mm256_setzero_si256();
                    _sum2 = _mm256_setzero_si256();
                    _sum3 = _mm256_setzero_si256();
#else
                    _sum0 = _mm_setzero_si128();
                    _sum1 = _mm_setzero_si128();
                    _sum2 = _mm_setzero_si128();
                    _sum3 = _mm_setzero_si128();
                    _sum4 = _mm_setzero_si128();
                    _sum5 = _mm_setzero_si128();
                    _sum6 = _mm_setzero_si128();
                    _sum7 = _mm_setzero_si128();
#endif
                }
                else
                {
#if __AVX2__
                    _sum0 = _mm256_loadu_si256((const __m256i*)outptr);
                    _sum1 = _mm256_loadu_si256((const __m256i*)(outptr + 8));
                    _sum2 = _mm256_loadu_si256((const __m256i*)(outptr + 16));
                    _sum3 = _mm256_loadu_si256((const __m256i*)(outptr + 24));
#else
                    _sum0 = _mm_load_si128((const __m128i*)outptr);
                    _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                    _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                    _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
                    _sum4 = _mm_load_si128((const __m128i*)(outptr + 16));
                    _sum5 = _mm_load_si128((const __m128i*)(outptr + 20));
                    _sum6 = _mm_load_si128((const __m128i*)(outptr + 24));
                    _sum7 = _mm_load_si128((const __m128i*)(outptr + 28));
#endif
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
#if __AVX2__
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pB);
                    __m256i _pA0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pA), _pA, 1);
                    __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

#if __AVXVNNI__ || __AVX512VNNI__
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _pA1, _pB0);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _pA1, _pB1);
#else
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA1, _pB0));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA1, _pB1));
#endif
#else // __AVX2__
                    __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                    __m128i _pB1 = _mm_loadu_si128((const __m128i*)(pB + 8));
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB2 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    __m128i _pB3 = _mm_shuffle_epi32(_pB1, _MM_SHUFFLE(0, 3, 2, 1));

#if __XOP__
                    _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                    _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                    _sum2 = _mm_maddd_epi16(_pA0, _pB2, _sum2);
                    _sum3 = _mm_maddd_epi16(_pA0, _pB3, _sum3);
                    _sum4 = _mm_maddd_epi16(_pA1, _pB0, _sum4);
                    _sum5 = _mm_maddd_epi16(_pA1, _pB1, _sum5);
                    _sum6 = _mm_maddd_epi16(_pA1, _pB2, _sum6);
                    _sum7 = _mm_maddd_epi16(_pA1, _pB3, _sum7);
#else
                    _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA0, _pB2));
                    _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA0, _pB3));
                    _sum4 = _mm_add_epi32(_sum4, _mm_madd_epi16(_pA1, _pB0));
                    _sum5 = _mm_add_epi32(_sum5, _mm_madd_epi16(_pA1, _pB1));
                    _sum6 = _mm_add_epi32(_sum6, _mm_madd_epi16(_pA1, _pB2));
                    _sum7 = _mm_add_epi32(_sum7, _mm_madd_epi16(_pA1, _pB3));
#endif
#endif // __AVX2__

                    pA += 8;
                    pB += 16;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                    __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __AVX2__
                    __m256i _pA0 = _mm256_cvtepi16_epi32(_pA);
                    __m256i _pB0 = _mm256_cvtepi16_epi32(_pB);
                    __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                    __m256i _s0 = _mm256_mullo_epi32(_pA0, _pB0);
                    __m256i _s1 = _mm256_mullo_epi32(_pA0, _pB1);
                    __m256i _s2 = _mm256_mullo_epi32(_pA1, _pB0);
                    __m256i _s3 = _mm256_mullo_epi32(_pA1, _pB1);

                    _sum0 = _mm256_add_epi32(_sum0, _s0);
                    _sum1 = _mm256_add_epi32(_sum1, _s1);
                    _sum2 = _mm256_add_epi32(_sum2, _s2);
                    _sum3 = _mm256_add_epi32(_sum3, _s3);
#else // __AVX2__
#if __XOP__
                    __m128i _pA0 = _mm_unpacklo_epi16(_pA, _pA);
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                    __m128i _pB1 = _mm_unpackhi_epi16(_pB, _pB);
                    __m128i _pB2 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    __m128i _pB3 = _mm_shuffle_epi32(_pB1, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_maccd_epi16(_pA0, _pB0, _sum0);
                    _sum1 = _mm_maccd_epi16(_pA0, _pB1, _sum1);
                    _sum2 = _mm_maccd_epi16(_pA0, _pB2, _sum2);
                    _sum3 = _mm_maccd_epi16(_pA0, _pB3, _sum3);
                    _sum4 = _mm_maccd_epi16(_pA1, _pB0, _sum4);
                    _sum5 = _mm_maccd_epi16(_pA1, _pB1, _sum5);
                    _sum6 = _mm_maccd_epi16(_pA1, _pB2, _sum6);
                    _sum7 = _mm_maccd_epi16(_pA1, _pB3, _sum7);
#else
                    __m128i _pA0 = _pA;
                    __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));
                    __m128i _pB01 = _pB;
                    __m128i _pB23 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                    __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB01);
                    __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB01);
                    __m128i _sl1 = _mm_mullo_epi16(_pA0, _pB23);
                    __m128i _sh1 = _mm_mulhi_epi16(_pA0, _pB23);
                    __m128i _sl2 = _mm_mullo_epi16(_pA1, _pB01);
                    __m128i _sh2 = _mm_mulhi_epi16(_pA1, _pB01);
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
#endif
#endif // __AVX2__

                    pA += 4;
                    pB += 8;
                }

#if __AVX2__
                if (k_end)
                {
                    // from
                    //      00 11 22 33 04 15 26 37
                    //      01 12 23 30 05 16 27 34
                    //      20 31 02 13 24 35 06 17
                    //      21 32 03 10 25 36 07 14
                    // to
                    //      00 10 20 30 04 14 24 34
                    //      01 11 21 31 05 15 25 35
                    //      02 12 22 32 06 16 26 36
                    //      03 13 23 33 07 17 27 37
                    {
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum3);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum3);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum1);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _tmp0 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                        _tmp1 = _mm256_permute2x128_si256(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                        _tmp2 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                        _tmp3 = _mm256_permute2x128_si256(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));
                        _sum0 = _tmp0;
                        _sum1 = _tmp1;
                        _sum2 = _tmp2;
                        _sum3 = _tmp3;
                    }
                }

                _mm256_storeu_si256((__m256i*)outptr, _sum0);
                _mm256_storeu_si256((__m256i*)(outptr + 8), _sum1);
                _mm256_storeu_si256((__m256i*)(outptr + 16), _sum2);
                _mm256_storeu_si256((__m256i*)(outptr + 24), _sum3);
                outptr += 32;
#else  // __AVX2__
                if (k_end)
                {
                    // from
                    //      00 11 22 33  04 15 26 37
                    //      01 12 23 30  05 16 27 34
                    //      20 31 02 13  24 35 06 17
                    //      21 32 03 10  25 36 07 14
                    // to
                    //      00 10 20 30  04 14 24 34
                    //      01 11 21 31  05 15 25 35
                    //      02 12 22 32  06 16 26 36
                    //      03 13 23 33  07 17 27 37
                    {
                        _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum6 = _mm_shuffle_epi32(_sum6, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                        __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum6);
                        __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum6);
                        __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum7);
                        __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum7);
                        __m128i _tmp4 = _mm_unpacklo_epi32(_sum4, _sum2);
                        __m128i _tmp5 = _mm_unpackhi_epi32(_sum4, _sum2);
                        __m128i _tmp6 = _mm_unpacklo_epi32(_sum5, _sum3);
                        __m128i _tmp7 = _mm_unpackhi_epi32(_sum5, _sum3);
                        _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp4);
                        _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp4);
                        _sum2 = _mm_unpacklo_epi64(_tmp5, _tmp1);
                        _sum3 = _mm_unpackhi_epi64(_tmp5, _tmp1);
                        _sum4 = _mm_unpacklo_epi64(_tmp2, _tmp6);
                        _sum5 = _mm_unpackhi_epi64(_tmp2, _tmp6);
                        _sum6 = _mm_unpacklo_epi64(_tmp7, _tmp3);
                        _sum7 = _mm_unpackhi_epi64(_tmp7, _tmp3);
                        _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum5 = _mm_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                    }
                }

                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                _mm_store_si128((__m128i*)(outptr + 8), _sum2);
                _mm_store_si128((__m128i*)(outptr + 12), _sum3);
                _mm_store_si128((__m128i*)(outptr + 16), _sum4);
                _mm_store_si128((__m128i*)(outptr + 20), _sum5);
                _mm_store_si128((__m128i*)(outptr + 24), _sum6);
                _mm_store_si128((__m128i*)(outptr + 28), _sum7);
                outptr += 32;
#endif // __AVX2__
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

#if __XOP__
                    _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                    _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                    _sum2 = _mm_maddd_epi16(_pA1, _pB0, _sum2);
                    _sum3 = _mm_maddd_epi16(_pA1, _pB1, _sum3);
#else
                    _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA1, _pB0));
                    _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA1, _pB1));
#endif

                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                    __m128i _pB = _mm_castpd_si128(_mm_load1_pd((const double*)pB));
#if __XOP__
                    __m128i _pA0 = _mm_unpacklo_epi16(_pA, _pA);
                    __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                    _sum0 = _mm_maccd_epi16(_pA0, _pB0, _sum0);
                    _sum1 = _mm_maccd_epi16(_pA0, _pB1, _sum1);
                    _sum2 = _mm_maccd_epi16(_pA1, _pB0, _sum2);
                    _sum3 = _mm_maccd_epi16(_pA1, _pB1, _sum3);
#else
                    __m128i _pA0 = _pA;
                    __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));
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

                if (k_end)
                {
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
                        _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum3);
                        __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum3);
                        __m128i _tmp2 = _mm_unpacklo_epi32(_sum2, _sum1);
                        __m128i _tmp3 = _mm_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }
                }

                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                _mm_store_si128((__m128i*)(outptr + 8), _sum2);
                _mm_store_si128((__m128i*)(outptr + 12), _sum3);
                outptr += 16;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB0 = _mm_castpd_si128(_mm_load1_pd((const double*)pB));
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));

#if __XOP__
                    _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                    _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                    _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif

                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                    __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __XOP__
                    _pA = _mm_unpacklo_epi16(_pA, _pA);
                    __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                    __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                    _sum0 = _mm_maccd_epi16(_pA, _pB0, _sum0);
                    _sum1 = _mm_maccd_epi16(_pA, _pB1, _sum1);
#else
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

                if (k_end)
                {
                    // from
                    //      00 11 20 31
                    //      01 10 21 30
                    // to
                    //      00 10 20 30
                    //      01 11 21 31
                    {
                        __m128i _tmp0 = _mm_shuffle_epi32(_sum0, _MM_SHUFFLE(3, 1, 2, 0));
                        __m128i _tmp1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(0, 2, 3, 1));
                        _sum0 = _mm_unpacklo_epi32(_tmp0, _tmp1);
                        _sum1 = _mm_unpackhi_epi32(_tmp0, _tmp1);
                        _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                    }
                }

                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                outptr += 8;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

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
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                    __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __XOP__
                    _sum0 = _mm_maddd_epi16(_pA, _pB, _sum0);
#else
                    _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB));
#endif

                    pA += 8;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                    __m128i _pB = _mm_set1_epi16(pB[0]);

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
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
            for (; jj + 15 < max_jj; jj += 16)
            {
                const short* pA = pAT;

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

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m512i _pA0 = _mm512_set1_epi32(((const int*)pA)[0]);
                    __m512i _pA1 = _mm512_set1_epi32(((const int*)pA)[1]);
                    __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _pA1, _pB0);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA1, _pB0));
#endif // __AVX512VNNI__

                    pA += 4;
                    pB += 32;
                }
                for (; kk < max_kk; kk++)
                {
                    __m512i _pA0 = _mm512_set1_epi32(pA[0]);
                    __m512i _pA1 = _mm512_set1_epi32(pA[1]);
                    __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                    __m512i _pB0 = _mm512_cvtepi16_epi32(_pB);

                    __m512i _s0 = _mm512_mullo_epi32(_pA0, _pB0);
                    __m512i _s1 = _mm512_mullo_epi32(_pA1, _pB0);
                    _sum0 = _mm512_add_epi32(_sum0, _s0);
                    _sum1 = _mm512_add_epi32(_sum1, _s1);

                    pA += 2;
                    pB += 16;
                }

                if (k_end)
                {
                    __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                    __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(1, 0, 1, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum0 = _mm512_shuffle_i32x4(_sum0, _sum0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_sum1, _sum1, _MM_SHUFFLE(3, 1, 2, 0));
                }

                _mm512_storeu_si512((__m512i*)outptr, _sum0);
                _mm512_storeu_si512((__m512i*)(outptr + 16), _sum1);
                outptr += 32;
            }
#endif // __AVX512F__
            for (; jj + 7 < max_jj; jj += 8)
            {
                const short* pA = pAT;

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
                    _sum0 = _mm_loadu_si128((const __m128i*)outptr);
                    _sum1 = _mm_loadu_si128((const __m128i*)(outptr + 4));
                    _sum2 = _mm_loadu_si128((const __m128i*)(outptr + 8));
                    _sum3 = _mm_loadu_si128((const __m128i*)(outptr + 12));
#endif
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
#if __AVX2__
                    __m256i _pA0 = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pA));
                    __m256i _pA1 = _mm256_castps_si256(_mm256_broadcast_ss((const float*)(pA + 2)));
                    __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pB);

                    // vs2019 internal compiler error with avx512 vnni intrinsics here
                    // fallback to avx2 madd anyway as a workaround   --- nihui
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA1, _pB0));
#else  // __AVX2__
                    __m128i _pA0 = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                    __m128i _pA1 = _mm_castps_si128(_mm_load1_ps((const float*)(pA + 2)));
                    __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                    __m128i _pB1 = _mm_loadu_si128((const __m128i*)(pB + 8));
                    _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                    _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA1, _pB0));
                    _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA1, _pB1));
#endif // __AVX2__

                    pA += 4;
                    pB += 16;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pB = _mm_load_si128((const __m128i*)pB);
#if __AVX2__
                    __m256i _pA0 = _mm256_set1_epi32(pA[0]);
                    __m256i _pA1 = _mm256_set1_epi32(pA[1]);
                    __m256i _pB0 = _mm256_cvtepi16_epi32(_pB);

                    __m256i _s0 = _mm256_mullo_epi32(_pA0, _pB0);
                    __m256i _s1 = _mm256_mullo_epi32(_pA1, _pB0);
                    _sum0 = _mm256_add_epi32(_sum0, _s0);
                    _sum1 = _mm256_add_epi32(_sum1, _s1);
#else  // __AVX2__
                    __m128i _pA0 = _mm_set1_epi16(pA[0]);
                    __m128i _pA1 = _mm_set1_epi16(pA[1]);

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
                if (k_end)
                {
                    __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum1);
                    __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum1);
                    _sum0 = _mm256_permute2x128_si256(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
                    _sum1 = _mm256_permute2x128_si256(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));
                }

                _mm256_storeu_si256((__m256i*)outptr, _sum0);
                _mm256_storeu_si256((__m256i*)(outptr + 8), _sum1);
                outptr += 16;
#else  // __AVX2__
                if (k_end)
                {
                    __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum2);
                    __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum2);
                    __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum3);
                    __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum3);
                    _sum0 = _tmp0;
                    _sum1 = _tmp1;
                    _sum2 = _tmp2;
                    _sum3 = _tmp3;
                }

                _mm_storeu_si128((__m128i*)outptr, _sum0);
                _mm_storeu_si128((__m128i*)(outptr + 4), _sum1);
                _mm_storeu_si128((__m128i*)(outptr + 8), _sum2);
                _mm_storeu_si128((__m128i*)(outptr + 12), _sum3);
                outptr += 16;
#endif // __AVX2__
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

                __m128i _sum0;
                __m128i _sum1;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_si128();
                    _sum1 = _mm_setzero_si128();
                }
                else
                {
                    _sum0 = _mm_loadu_si128((const __m128i*)outptr);
                    _sum1 = _mm_loadu_si128((const __m128i*)(outptr + 4));
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA0 = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                    __m128i _pA1 = _mm_castps_si128(_mm_load1_ps((const float*)(pA + 2)));
                    __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                    _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB));
                    _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA1, _pB));

                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA0 = _mm_set1_epi16(pA[0]);
                    __m128i _pA1 = _mm_set1_epi16(pA[1]);
                    __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);
                    __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB);
                    __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB);
                    __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB);
                    __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB);
                    __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                    __m128i _s1 = _mm_unpacklo_epi16(_sl1, _sh1);
                    _sum0 = _mm_add_epi32(_sum0, _s0);
                    _sum1 = _mm_add_epi32(_sum1, _s1);
                    pA += 2;
                    pB += 4;
                }

                if (k_end)
                {
                    __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                    __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum1);
                    _sum0 = _tmp0;
                    _sum1 = _tmp1;
                }

                _mm_storeu_si128((__m128i*)outptr, _sum0);
                _mm_storeu_si128((__m128i*)(outptr + 4), _sum1);
                outptr += 2 * 4;
            }
#endif // __SSE2__
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;

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

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum00 += pA[0] * pB[0];
                    sum00 += pA[1] * pB[1];
                    sum01 += pA[2] * pB[0];
                    sum01 += pA[3] * pB[1];
                    sum10 += pA[0] * pB[2];
                    sum10 += pA[1] * pB[3];
                    sum11 += pA[2] * pB[2];
                    sum11 += pA[3] * pB[3];

                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum01 += pA[1] * pB[0];
                    sum10 += pA[0] * pB[1];
                    sum11 += pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                }

                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
                outptr += 2 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

                int sum0 = 0;
                int sum1 = 0;

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

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[1];
                    sum1 += pA[2] * pB[0];
                    sum1 += pA[3] * pB[1];
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
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
        }
    }
    for (; ii < max_ii; ii++)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
            for (; jj + 15 < max_jj; jj += 16)
            {
                const short* pA = pAT;

                __m512i _sum0;

                if (k == 0)
                {
                    _sum0 = _mm512_setzero_si512();
                }
                else
                {
                    _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m512i _pA0 = _mm512_set1_epi32(((const int*)pA)[0]);
                    __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
#endif

                    pA += 2;
                    pB += 32;
                }
                for (; kk < max_kk; kk++)
                {
                    __m512i _pA = _mm512_set1_epi32(pA[0]);
                    __m512i _pB0 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)pB));

                    __m512i _s0 = _mm512_mullo_epi32(_pA, _pB0);
                    _sum0 = _mm512_add_epi32(_sum0, _s0);

                    pA += 1;
                    pB += 16;
                }

                _mm512_storeu_si512((__m512i*)outptr, _sum0);
                outptr += 16;
            }
#endif // __AVX512F__
            for (; jj + 7 < max_jj; jj += 8)
            {
                const short* pA = pAT;

                __m128i _sum0;
                __m128i _sum1;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_si128();
                    _sum1 = _mm_setzero_si128();
                }
                else
                {
                    _sum0 = _mm_loadu_si128((const __m128i*)outptr);
                    _sum1 = _mm_loadu_si128((const __m128i*)(outptr + 4));
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                    __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                    __m128i _pB1 = _mm_loadu_si128((const __m128i*)(pB + 8));
                    _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));

                    pA += 2;
                    pB += 16;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = _mm_set1_epi16(pA[0]);
                    __m128i _pB = _mm_load_si128((const __m128i*)pB);
                    __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                    __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                    __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                    __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);
                    _sum0 = _mm_add_epi32(_sum0, _s0);
                    _sum1 = _mm_add_epi32(_sum1, _s1);
                    pA += 1;
                    pB += 8;
                }

                _mm_storeu_si128((__m128i*)outptr, _sum0);
                _mm_storeu_si128((__m128i*)(outptr + 4), _sum1);
                outptr += 8;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

                __m128i _sum0;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_si128();
                }
                else
                {
                    _sum0 = _mm_loadu_si128((const __m128i*)outptr);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                    __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                    _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB));

                    pA += 2;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = _mm_set1_epi16(pA[0]);
                    __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);
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
                const short* pA = pAT;

                int sum0 = 0;
                int sum1 = 0;

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

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[1];
                    sum1 += pA[0] * pB[2];
                    sum1 += pA[1] * pB[3];
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
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
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

                int sum = 0;

                if (k == 0)
                {
                    sum = 0;
                }
                else
                {
                    sum = outptr[0];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum += pA[0] * pB[0];
                    pA += 1;
                    pB += 1;
                }

                outptr[0] = sum;
                outptr += 1;
            }
        }
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve M
    {
        int tile_size = (int)sqrt((float)l2_cache_size_int8 / 3);

#if __AVX512F__
        TILE_M = std::max(16, tile_size / 16 * 16);
#elif __AVX2__
        TILE_M = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
        TILE_M = std::max(4, tile_size / 4 * 4);
#else
        TILE_M = std::max(2, tile_size / 2 * 2);
#endif

        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __AVX512F__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX2__
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
#elif __AVX2__
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __SSE2__
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    // solve K
    {
        int tile_size = (int)(sqrt((float)l2_cache_size_int8) - TILE_M);

#if __AVX512F__
        TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX2__
        TILE_K = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __AVX512F__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 15) / 16 * 16);
#elif __AVX2__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __SSE2__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int tile_size = (int)((l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M * 2 + TILE_K));

#if __SSE2__
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __SSE2__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static inline void conv3x3s1_winograd23_transform_kernel_tile_int8(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const signed char ktm[4][3] = {
    //     {2, 0, 0},
    //     {1, 1, 1},
    //     {1, -1, 1},
    //     {0, 0, 2}
    // };

    short* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            short tmp[4][3];

            const signed char* k0 = (const signed char*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                signed char r0 = k0[0];
                signed char r1 = k0[1];
                signed char r2 = k0[2];

                tmp[0][m] = r0 * 2;
                tmp[1][m] = r0 + r1 + r2;
                tmp[2][m] = r0 - r1 + r2;
                tmp[3][m] = r2 * 2;

                k0 += 3;
            }

            for (int m = 0; m < 4; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];

                short z0 = r0 * 2;
                short z1 = r0 + r1 + r2;
                short z2 = r0 - r1 + r2;
                short z3 = r2 * 2;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp += 4;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        conv3x3s1_winograd23_transform_kernel_int8_avx2(kernel, AT, inch, outch, opt);
        return;
    }
#endif
#endif

    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, 2u, (Allocator*)0);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, (Allocator*)0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd23_transform_kernel_tile_int8(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile_int8(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd23_transform_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const signed char itm[4][4] = {
    //     {1,  0, -1,  0},
    //     {0,  1,  1,  0},
    //     {0, -1,  1,  0},
    //     {0, -1,  0,  1}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w - 1) / 2;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __SSE2__
#if __AVX512F__
    nn_max_kk = max_kk / 16;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = ppkk * 16;

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        short tmp[4][4][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel((k + kk) / elempack).row<const signed char>(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m256i _r0 = _mm256_setzero_si256();
                __m256i _r1 = _mm256_setzero_si256();
                __m256i _r2 = _mm256_setzero_si256();
                __m256i _r3 = _mm256_setzero_si256();

                if (ti * 2 + m < h)
                {
                    if (elempack == 16)
                    {
                        _r0 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)r0));
                        if (tj * 2 + 1 < w) _r1 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(r0 + 16)));
                        if (tj * 2 + 2 < w) _r2 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(r0 + 32)));
                        if (tj * 2 + 3 < w) _r3 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(r0 + 48)));
                    }
                    if (elempack == 8)
                    {
                        const signed char* r1 = r0 + N;

                        _r0 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)r0), _mm_loadl_epi64((const __m128i*)r1)));
                        if (tj * 2 + 1 < w) _r1 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r0 + 8)), _mm_loadl_epi64((const __m128i*)(r1 + 8))));
                        if (tj * 2 + 2 < w) _r2 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r0 + 16)), _mm_loadl_epi64((const __m128i*)(r1 + 16))));
                        if (tj * 2 + 3 < w) _r3 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r0 + 24)), _mm_loadl_epi64((const __m128i*)(r1 + 24))));
                    }
                    if (elempack == 1)
                    {
                        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                        _r0 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)r0, 1)));
                        if (tj * 2 + 1 < w) _r1 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0 + 1), 1)));
                        if (tj * 2 + 2 < w) _r2 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0 + 2), 1)));
                        if (tj * 2 + 3 < w) _r3 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0 + 3), 1)));
                    }
                }

                __m256i _tmp0 = _mm256_sub_epi16(_r0, _r2);
                __m256i _tmp1 = _mm256_add_epi16(_r1, _r2);
                __m256i _tmp2 = _mm256_sub_epi16(_r2, _r1);
                __m256i _tmp3 = _mm256_sub_epi16(_r3, _r1);

                _mm256_store_si256((__m256i*)tmp[0][m], _tmp0);
                _mm256_store_si256((__m256i*)tmp[1][m], _tmp1);
                _mm256_store_si256((__m256i*)tmp[2][m], _tmp2);
                _mm256_store_si256((__m256i*)tmp[3][m], _tmp3);

                r0 += w * elempack;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj * 16;
            short* p1 = p0 + max_jj * 16;
            short* p2 = p0 + max_jj * 16 * 2;
            short* p3 = p0 + max_jj * 16 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m256i _r0 = _mm256_load_si256((const __m256i*)tmp[m][0]);
                __m256i _r1 = _mm256_load_si256((const __m256i*)tmp[m][1]);
                __m256i _r2 = _mm256_load_si256((const __m256i*)tmp[m][2]);
                __m256i _r3 = _mm256_load_si256((const __m256i*)tmp[m][3]);

                __m256i _tmp0 = _mm256_sub_epi16(_r0, _r2);
                __m256i _tmp1 = _mm256_add_epi16(_r1, _r2);
                __m256i _tmp2 = _mm256_sub_epi16(_r2, _r1);
                __m256i _tmp3 = _mm256_sub_epi16(_r3, _r1);

                _mm256_store_si256((__m256i*)p0, _tmp0);
                _mm256_store_si256((__m256i*)p1, _tmp1);
                _mm256_store_si256((__m256i*)p2, _tmp2);
                _mm256_store_si256((__m256i*)p3, _tmp3);

                p0 += max_jj * 4 * 16;
                p1 += max_jj * 4 * 16;
                p2 += max_jj * 4 * 16;
                p3 += max_jj * 4 * 16;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 16;
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
#else // __AVX512F__
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
#endif // __AVX512F__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

#ifdef _MSC_VER
        __declspec(align(32))
#else
        __attribute__((aligned(32)))
#endif
        short tmp[4][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel((k + kk) / elempack).row<const signed char>(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m128i _r0 = _mm_setzero_si128();
                __m128i _r1 = _mm_setzero_si128();
                __m128i _r2 = _mm_setzero_si128();
                __m128i _r3 = _mm_setzero_si128();

                if (ti * 2 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = _mm_loadl_epi64((const __m128i*)r0);
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
                        if (tj * 2 + 1 < w)
                        {
                            _r1 = _mm_loadl_epi64((const __m128i*)(r0 + 8));
                            _r1 = _mm_unpacklo_epi8(_r1, _mm_cmpgt_epi8(_mm_setzero_si128(), _r1));
                        }
                        if (tj * 2 + 2 < w)
                        {
                            _r2 = _mm_loadl_epi64((const __m128i*)(r0 + 16));
                            _r2 = _mm_unpacklo_epi8(_r2, _mm_cmpgt_epi8(_mm_setzero_si128(), _r2));
                        }
                        if (tj * 2 + 3 < w)
                        {
                            _r3 = _mm_loadl_epi64((const __m128i*)(r0 + 24));
                            _r3 = _mm_unpacklo_epi8(_r3, _mm_cmpgt_epi8(_mm_setzero_si128(), _r3));
                        }
                    }
                    if (elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                        _r0 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0, _vindex, 1)));
                        if (tj * 2 + 1 < w) _r1 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(r0 + 1), _vindex, 1)));
                        if (tj * 2 + 2 < w) _r2 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(r0 + 2), _vindex, 1)));
                        if (tj * 2 + 3 < w) _r3 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(r0 + 3), _vindex, 1)));
#else
                        __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                        __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                        __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0, _vindex, 1), _sindex88);
                        _r0 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1)));
                        if (tj * 2 + 1 < w)
                        {
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(r0 + 1), _vindex, 1), _sindex88);
                            _r1 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1)));
                        }
                        if (tj * 2 + 2 < w)
                        {
                            __m256i _val2_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(r0 + 2), _vindex, 1), _sindex88);
                            _r2 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val2_32, 0), _mm256_extracti128_si256(_val2_32, 1)));
                        }
                        if (tj * 2 + 3 < w)
                        {
                            __m256i _val3_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(r0 + 3), _vindex, 1), _sindex88);
                            _r3 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val3_32, 0), _mm256_extracti128_si256(_val3_32, 1)));
                        }
#endif // __AVX512F__
#else  // __AVX2__
                        const signed char* r1 = r0 + N;
                        const signed char* r2 = r0 + N * 2;
                        const signed char* r3 = r0 + N * 3;
                        const signed char* r4 = r0 + N * 4;
                        const signed char* r5 = r0 + N * 5;
                        const signed char* r6 = r0 + N * 6;
                        const signed char* r7 = r0 + N * 7;

                        __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0);
                        __m128i _t1 = _mm_loadl_epi64((const __m128i*)r1);
                        __m128i _t2 = _mm_loadl_epi64((const __m128i*)r2);
                        __m128i _t3 = _mm_loadl_epi64((const __m128i*)r3);
                        __m128i _t4 = _mm_loadl_epi64((const __m128i*)r4);
                        __m128i _t5 = _mm_loadl_epi64((const __m128i*)r5);
                        __m128i _t6 = _mm_loadl_epi64((const __m128i*)r6);
                        __m128i _t7 = _mm_loadl_epi64((const __m128i*)r7);

                        __m128i _t01 = _mm_unpacklo_epi8(_t0, _t1);
                        __m128i _t23 = _mm_unpacklo_epi8(_t2, _t3);
                        __m128i _t45 = _mm_unpacklo_epi8(_t4, _t5);
                        __m128i _t67 = _mm_unpacklo_epi8(_t6, _t7);
                        _t0 = _mm_unpacklo_epi16(_t01, _t23);
                        _t1 = _mm_unpacklo_epi16(_t45, _t67);
                        _t2 = _mm_unpacklo_epi32(_t0, _t1);
                        _t3 = _mm_unpackhi_epi32(_t0, _t1);

                        __m128i _extt2 = _mm_cmpgt_epi8(_mm_setzero_si128(), _t2);
                        __m128i _extt3 = _mm_cmpgt_epi8(_mm_setzero_si128(), _t3);

                        _r0 = _mm_unpacklo_epi8(_t2, _extt2);
                        if (tj * 2 + 1 < w) _r1 = _mm_unpackhi_epi8(_t2, _extt2);
                        if (tj * 2 + 2 < w) _r2 = _mm_unpacklo_epi8(_t3, _extt3);
                        if (tj * 2 + 3 < w) _r3 = _mm_unpackhi_epi8(_t3, _extt3);
#endif // __AVX2__
                    }
                }

                __m128i _tmp0 = _mm_sub_epi16(_r0, _r2);
                __m128i _tmp1 = _mm_add_epi16(_r1, _r2);
                __m128i _tmp2 = _mm_sub_epi16(_r2, _r1);
                __m128i _tmp3 = _mm_sub_epi16(_r3, _r1);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                // old gcc breaks stack variable alignement
                // ref https://gcc.gnu.org/bugzilla/show_bug.cgi?id=16660
                _mm_storeu_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_storeu_si128((__m128i*)tmp[1][m], _tmp1);
                _mm_storeu_si128((__m128i*)tmp[2][m], _tmp2);
                _mm_storeu_si128((__m128i*)tmp[3][m], _tmp3);
#else
                _mm_store_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_store_si128((__m128i*)tmp[1][m], _tmp1);
                _mm_store_si128((__m128i*)tmp[2][m], _tmp2);
                _mm_store_si128((__m128i*)tmp[3][m], _tmp3);
#endif

                r0 += w * elempack;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj * 8;
            short* p1 = p0 + max_jj * 8;
            short* p2 = p0 + max_jj * 8 * 2;
            short* p3 = p0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128i _r0 = _mm_loadu_si128((const __m128i*)tmp[m][0]);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)tmp[m][1]);
                __m128i _r2 = _mm_loadu_si128((const __m128i*)tmp[m][2]);
                __m128i _r3 = _mm_loadu_si128((const __m128i*)tmp[m][3]);
#else
                __m128i _r0 = _mm_load_si128((const __m128i*)tmp[m][0]);
                __m128i _r1 = _mm_load_si128((const __m128i*)tmp[m][1]);
                __m128i _r2 = _mm_load_si128((const __m128i*)tmp[m][2]);
                __m128i _r3 = _mm_load_si128((const __m128i*)tmp[m][3]);
#endif

                __m128i _tmp0 = _mm_sub_epi16(_r0, _r2);
                __m128i _tmp1 = _mm_add_epi16(_r1, _r2);
                __m128i _tmp2 = _mm_sub_epi16(_r2, _r1);
                __m128i _tmp3 = _mm_sub_epi16(_r3, _r1);

                _mm_store_si128((__m128i*)p0, _tmp0);
                _mm_store_si128((__m128i*)p1, _tmp1);
                _mm_store_si128((__m128i*)p2, _tmp2);
                _mm_store_si128((__m128i*)p3, _tmp3);

                p0 += max_jj * 4 * 8;
                p1 += max_jj * 4 * 8;
                p2 += max_jj * 4 * 8;
                p3 += max_jj * 4 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __SSE2__
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __SSE2__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        short tmp[4][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel(k + kk).row<const signed char>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                signed char r00 = 0;
                signed char r01 = 0;
                signed char r10 = 0;
                signed char r11 = 0;
                signed char r20 = 0;
                signed char r21 = 0;
                signed char r30 = 0;
                signed char r31 = 0;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 2 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 2 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 2 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                    }
                }

                tmp[0][m][0] = r00 - r20;
                tmp[0][m][1] = r01 - r21;
                tmp[1][m][0] = r10 + r20;
                tmp[1][m][1] = r11 + r21;
                tmp[2][m][0] = r20 - r10;
                tmp[2][m][1] = r21 - r11;
                tmp[3][m][0] = r30 - r10;
                tmp[3][m][1] = r31 - r11;

                r0 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj * 2;
            short* p1 = p0 + max_jj * 2;
            short* p2 = p0 + max_jj * 2 * 2;
            short* p3 = p0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                short r00 = tmp[m][0][0];
                short r01 = tmp[m][0][1];
                short r10 = tmp[m][1][0];
                short r11 = tmp[m][1][1];
                short r20 = tmp[m][2][0];
                short r21 = tmp[m][2][1];
                short r30 = tmp[m][3][0];
                short r31 = tmp[m][3][1];

                p0[0] = r00 - r20;
                p0[1] = r01 - r21;
                p1[0] = r10 + r20;
                p1[1] = r11 + r21;
                p2[0] = r20 - r10;
                p2[1] = r21 - r11;
                p3[0] = r30 - r10;
                p3[1] = r31 - r11;

                p0 += max_jj * 4 * 2;
                p1 += max_jj * 4 * 2;
                p2 += max_jj * 4 * 2;
                p3 += max_jj * 4 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        short tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0123 = bottom_blob.channel(k + kk).row<const signed char>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                signed char r0 = 0;
                signed char r1 = 0;
                signed char r2 = 0;
                signed char r3 = 0;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 2 + 1 < w) r1 = r0123[1];
                        if (tj * 2 + 2 < w) r2 = r0123[2];
                        if (tj * 2 + 3 < w) r3 = r0123[3];
                    }
                }

                tmp[0][m] = r0 - r2;
                tmp[1][m] = r1 + r2;
                tmp[2][m] = r2 - r1;
                tmp[3][m] = r3 - r1;

                r0123 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj;
            short* p1 = p0 + max_jj;
            short* p2 = p0 + max_jj * 2;
            short* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];
                short r3 = tmp[m][3];

                p0[0] = r0 - r2;
                p1[0] = r1 + r2;
                p2[0] = r2 - r1;
                p3[0] = r3 - r1;

                p0 += max_jj * 4;
                p1 += max_jj * 4;
                p2 += max_jj * 4;
                p3 += max_jj * 4;
            }
        }
    }
}

static inline void conv3x3s1_winograd23_transform_output_tile_int8(const Mat& top_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    // const int otm[2][4] = {
    //     {1,  1,  1,  0},
    //     {0,  1, -1,  1}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 1) / 2;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        int tmp[2][4][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj * 16;
            const int* r1 = r0 + max_jj * 16;
            const int* r2 = r0 + max_jj * 16 * 2;
            const int* r3 = r0 + max_jj * 16 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m512i _r0 = _mm512_load_si512((const __m512i*)r0);
                __m512i _r1 = _mm512_load_si512((const __m512i*)r1);
                __m512i _r2 = _mm512_load_si512((const __m512i*)r2);
                __m512i _r3 = _mm512_load_si512((const __m512i*)r3);

                __m512i _tmp0 = _mm512_add_epi32(_mm512_add_epi32(_r0, _r1), _r2);
                __m512i _tmp1 = _mm512_add_epi32(_mm512_sub_epi32(_r1, _r2), _r3);

                _mm512_store_si512((__m512i*)tmp[0][m], _tmp0);
                _mm512_store_si512((__m512i*)tmp[1][m], _tmp1);

                r0 += max_jj * 4 * 16;
                r1 += max_jj * 4 * 16;
                r2 += max_jj * 4 * 16;
                r3 += max_jj * 4 * 16;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __m512i _r0 = _mm512_load_si512((const __m512i*)tmp[m][0]);
                __m512i _r1 = _mm512_load_si512((const __m512i*)tmp[m][1]);
                __m512i _r2 = _mm512_load_si512((const __m512i*)tmp[m][2]);
                __m512i _r3 = _mm512_load_si512((const __m512i*)tmp[m][3]);

                __m512i _tmp0 = _mm512_add_epi32(_mm512_add_epi32(_r0, _r1), _r2);
                __m512i _tmp1 = _mm512_add_epi32(_mm512_sub_epi32(_r1, _r2), _r3);

                _tmp0 = _mm512_srai_epi32(_tmp0, 2);
                _tmp1 = _mm512_srai_epi32(_tmp1, 2);

                if (out_elempack == 16)
                {
                    _mm512_store_si512((__m512i*)outptr0, _tmp0);
                    if (tj * 2 + 1 < outw)
                    {
                        _mm512_store_si512((__m512i*)(outptr0 + 16), _tmp1);
                    }
                }
                if (out_elempack == 8)
                {
                    int* outptr1 = outptr0 + N;

                    _mm256_store_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_tmp0, 0));
                    _mm256_store_si256((__m256i*)outptr1, _mm512_extracti32x8_epi32(_tmp0, 1));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm256_store_si256((__m256i*)(outptr0 + 8), _mm512_extracti32x8_epi32(_tmp1, 0));
                        _mm256_store_si256((__m256i*)(outptr1 + 8), _mm512_extracti32x8_epi32(_tmp1, 1));
                    }
                }
                if (out_elempack == 4)
                {
                    int* outptr1 = outptr0 + N;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;

                    _mm_store_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_tmp0, 0));
                    _mm_store_si128((__m128i*)outptr1, _mm512_extracti32x4_epi32(_tmp0, 1));
                    _mm_store_si128((__m128i*)outptr2, _mm512_extracti32x4_epi32(_tmp0, 2));
                    _mm_store_si128((__m128i*)outptr3, _mm512_extracti32x4_epi32(_tmp0, 3));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm_store_si128((__m128i*)(outptr0 + 4), _mm512_extracti32x4_epi32(_tmp1, 0));
                        _mm_store_si128((__m128i*)(outptr1 + 4), _mm512_extracti32x4_epi32(_tmp1, 1));
                        _mm_store_si128((__m128i*)(outptr2 + 4), _mm512_extracti32x4_epi32(_tmp1, 2));
                        _mm_store_si128((__m128i*)(outptr3 + 4), _mm512_extracti32x4_epi32(_tmp1, 3));
                    }
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                    _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                    _mm512_i32scatter_epi32(outptr0, _vindex, _tmp0, sizeof(int));
                    if (tj * 2 + 1 < outw) _mm512_i32scatter_epi32(outptr0 + 1, _vindex, _tmp1, sizeof(int));
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
#ifdef _MSC_VER
        __declspec(align(32))
#else
        __attribute__((aligned(32)))
#endif
        int tmp[2][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj * 8;
            const int* r1 = r0 + max_jj * 8;
            const int* r2 = r0 + max_jj * 8 * 2;
            const int* r3 = r0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m256i _r0 = _mm256_load_si256((const __m256i*)r0);
                __m256i _r1 = _mm256_load_si256((const __m256i*)r1);
                __m256i _r2 = _mm256_load_si256((const __m256i*)r2);
                __m256i _r3 = _mm256_load_si256((const __m256i*)r3);

                __m256i _tmp0 = _mm256_add_epi32(_mm256_add_epi32(_r0, _r1), _r2);
                __m256i _tmp1 = _mm256_add_epi32(_mm256_sub_epi32(_r1, _r2), _r3);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm256_storeu_si256((__m256i*)tmp[0][m], _tmp0);
                _mm256_storeu_si256((__m256i*)tmp[1][m], _tmp1);
#else
                _mm256_store_si256((__m256i*)tmp[0][m], _tmp0);
                _mm256_store_si256((__m256i*)tmp[1][m], _tmp1);
#endif

                r0 += max_jj * 4 * 8;
                r1 += max_jj * 4 * 8;
                r2 += max_jj * 4 * 8;
                r3 += max_jj * 4 * 8;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)tmp[m][0]);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)tmp[m][1]);
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)tmp[m][2]);
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)tmp[m][3]);
#else
                __m256i _r0 = _mm256_load_si256((const __m256i*)tmp[m][0]);
                __m256i _r1 = _mm256_load_si256((const __m256i*)tmp[m][1]);
                __m256i _r2 = _mm256_load_si256((const __m256i*)tmp[m][2]);
                __m256i _r3 = _mm256_load_si256((const __m256i*)tmp[m][3]);
#endif

                __m256i _tmp0 = _mm256_add_epi32(_mm256_add_epi32(_r0, _r1), _r2);
                __m256i _tmp1 = _mm256_add_epi32(_mm256_sub_epi32(_r1, _r2), _r3);

                _tmp0 = _mm256_srai_epi32(_tmp0, 2);
                _tmp1 = _mm256_srai_epi32(_tmp1, 2);

                if (out_elempack == 8)
                {
                    _mm256_store_si256((__m256i*)outptr0, _tmp0);
                    if (tj * 2 + 1 < outw)
                    {
                        _mm256_store_si256((__m256i*)(outptr0 + 8), _tmp1);
                    }
                }
                if (out_elempack == 4)
                {
                    int* outptr1 = outptr0 + N;

                    _mm_store_si128((__m128i*)outptr0, _mm256_extracti128_si256(_tmp0, 0));
                    _mm_store_si128((__m128i*)outptr1, _mm256_extracti128_si256(_tmp0, 1));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm_store_si128((__m128i*)(outptr0 + 4), _mm256_extracti128_si256(_tmp1, 0));
                        _mm_store_si128((__m128i*)(outptr1 + 4), _mm256_extracti128_si256(_tmp1, 1));
                    }
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
                    _mm256_i32scatter_epi32(outptr0, _vindex, _tmp0, sizeof(int));
                    if (tj * 2 + 1 < outw) _mm256_i32scatter_epi32(outptr0 + 1, _vindex, _tmp1, sizeof(int));
#else
                    int tmp0[8];
                    int tmp1[8];
                    _mm256_storeu_si256((__m256i*)tmp0, _tmp0);
                    _mm256_storeu_si256((__m256i*)tmp1, _tmp1);

                    int* outptr1 = outptr0 + N;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;
                    int* outptr4 = outptr0 + N * 4;
                    int* outptr5 = outptr0 + N * 5;
                    int* outptr6 = outptr0 + N * 6;
                    int* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];

                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
#endif // __AVX512F__
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        int tmp[2][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj * 4;
            const int* r1 = r0 + max_jj * 4;
            const int* r2 = r0 + max_jj * 4 * 2;
            const int* r3 = r0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m128i _r0 = _mm_load_si128((const __m128i*)r0);
                __m128i _r1 = _mm_load_si128((const __m128i*)r1);
                __m128i _r2 = _mm_load_si128((const __m128i*)r2);
                __m128i _r3 = _mm_load_si128((const __m128i*)r3);

                __m128i _tmp0 = _mm_add_epi32(_mm_add_epi32(_r0, _r1), _r2);
                __m128i _tmp1 = _mm_add_epi32(_mm_sub_epi32(_r1, _r2), _r3);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_storeu_si128((__m128i*)tmp[1][m], _tmp1);
#else
                _mm_store_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_store_si128((__m128i*)tmp[1][m], _tmp1);
#endif

                r0 += max_jj * 4 * 4;
                r1 += max_jj * 4 * 4;
                r2 += max_jj * 4 * 4;
                r3 += max_jj * 4 * 4;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128i _r0 = _mm_loadu_si128((const __m128i*)tmp[m][0]);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)tmp[m][1]);
                __m128i _r2 = _mm_loadu_si128((const __m128i*)tmp[m][2]);
                __m128i _r3 = _mm_loadu_si128((const __m128i*)tmp[m][3]);
#else
                __m128i _r0 = _mm_load_si128((const __m128i*)tmp[m][0]);
                __m128i _r1 = _mm_load_si128((const __m128i*)tmp[m][1]);
                __m128i _r2 = _mm_load_si128((const __m128i*)tmp[m][2]);
                __m128i _r3 = _mm_load_si128((const __m128i*)tmp[m][3]);
#endif

                __m128i _tmp0 = _mm_add_epi32(_mm_add_epi32(_r0, _r1), _r2);
                __m128i _tmp1 = _mm_add_epi32(_mm_sub_epi32(_r1, _r2), _r3);

                _tmp0 = _mm_srai_epi32(_tmp0, 2);
                _tmp1 = _mm_srai_epi32(_tmp1, 2);

                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)outptr0, _tmp0);
                    if (tj * 2 + 1 < outw) _mm_store_si128((__m128i*)(outptr0 + 4), _tmp1);
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                    _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(N));
                    _mm_i32scatter_epi32(outptr0, _vindex, _tmp0, sizeof(int));
                    if (tj * 2 + 1 < outw) _mm_i32scatter_epi32(outptr0 + 1, _vindex, _tmp1, sizeof(int));
#else
                    int tmp0[4];
                    int tmp1[4];
                    _mm_storeu_si128((__m128i*)tmp0, _tmp0);
                    _mm_storeu_si128((__m128i*)tmp1, _tmp1);

                    int* outptr1 = outptr0 + N;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];

                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
#endif // __AVX512F__
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        int tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj * 2;
            const int* r1 = r0 + max_jj * 2;
            const int* r2 = r0 + max_jj * 2 * 2;
            const int* r3 = r0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m][0] = r0[0] + r1[0] + r2[0];
                tmp[0][m][1] = r0[1] + r1[1] + r2[1];
                tmp[1][m][0] = r1[0] - r2[0] + r3[0];
                tmp[1][m][1] = r1[1] - r2[1] + r3[1];

                r0 += max_jj * 4 * 2;
                r1 += max_jj * 4 * 2;
                r2 += max_jj * 4 * 2;
                r3 += max_jj * 4 * 2;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                int tmp00 = tmp[m][0][0] + tmp[m][1][0] + tmp[m][2][0];
                int tmp01 = tmp[m][0][1] + tmp[m][1][1] + tmp[m][2][1];
                int tmp10 = tmp[m][1][0] - tmp[m][2][0] + tmp[m][3][0];
                int tmp11 = tmp[m][1][1] - tmp[m][2][1] + tmp[m][3][1];

                tmp00 = tmp00 >> 2;
                tmp01 = tmp01 >> 2;
                tmp10 = tmp10 >> 2;
                tmp11 = tmp11 >> 2;

                // if (out_elempack == 1)
                {
                    int* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        int tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj;
            const int* r1 = r0 + max_jj;
            const int* r2 = r0 + max_jj * 2;
            const int* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                int tmp0 = tmp[m][0] + tmp[m][1] + tmp[m][2];
                int tmp1 = tmp[m][1] - tmp[m][2] + tmp[m][3];

                tmp0 = tmp0 >> 2;
                tmp1 = tmp1 >> 2;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 2 + 1 < outw) outptr0[1] = tmp1;
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd23_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        conv3x3s1_winograd23_int8_avx512vnni(bottom_blob, top_blob, AT, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        conv3x3s1_winograd23_int8_avxvnni(bottom_blob, top_blob, AT, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        conv3x3s1_winograd23_int8_avx2(bottom_blob, top_blob, AT, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
    if (ncnn::cpu_support_x86_xop())
    {
        conv3x3s1_winograd23_int8_xop(bottom_blob, top_blob, AT, nT, opt);
        return;
    }
#endif
#endif

    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 2n+2, winograd F(2,3)
    int w_tiles = (outw + 1) / 2;
    int h_tiles = (outh + 1) / 2;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 16;

    // NCNN_LOGE("conv3x3s1_winograd23_int8 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 2u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd23_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 2u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd23_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                bool k_end = k + TILE_K >= K;

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, k_end);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile_int8(top_tile, top_blob, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_kernel_tile_int8(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const short ktm[6][3] = {
    //     {6, 0, 0},
    //     {-4, -4, -4},
    //     {-4, 4, -4},
    //     {1, 2, 4},
    //     {1, -2, 4},
    //     {0, 0, 6}
    // };

    short* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            short tmp[6][3];

            const signed char* k0 = (const signed char*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                signed char r0 = k0[0];
                signed char r1 = k0[1];
                signed char r2 = k0[2];

                tmp[0][m] = r0 * 6;
                tmp[1][m] = -r0 * 4 - r1 * 4 - r2 * 4;
                tmp[2][m] = -r0 * 4 + r1 * 4 - r2 * 4;
                tmp[3][m] = r0 + r1 * 2 + r2 * 4;
                tmp[4][m] = r0 - r1 * 2 + r2 * 4;
                tmp[5][m] = r2 * 6;

                k0 += 3;
            }

            for (int m = 0; m < 6; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];

                short z0 = r0 * 6;
                short z1 = -r0 * 4 - r1 * 4 - r2 * 4;
                short z2 = -r0 * 4 + r1 * 4 - r2 * 4;
                short z3 = r0 + r1 * 2 + r2 * 4;
                short z4 = r0 - r1 * 2 + r2 * 4;
                short z5 = r2 * 6;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp[4] = z4;
                ptmp[5] = z5;
                ptmp += 6;
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        conv3x3s1_winograd43_transform_kernel_int8_avx2(kernel, AT, inch, outch, opt);
        return;
    }
#endif
#endif

    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, 4u, (Allocator*)0);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 4u, (Allocator*)0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd43_transform_kernel_tile_int8(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile_int8(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[4][4] = {
    //     {4,  0, -5,  0, 1, 0},
    //     {0, -4, -4,  1, 1, 0},
    //     {0,  4, -4, -1, 1, 0},
    //     {0, -2, -1,  2, 1, 0},
    //     {0,  2, -1, -2, 1, 0},
    //     {0,  4,  0, -5, 0, 1}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 1) / 4;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __SSE2__
#if __AVX512F__
    nn_max_kk = max_kk / 16;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = ppkk * 16;

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        short tmp[6][6][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel((k + kk) / elempack).row<const signed char>(ti * 4) + (tj * 4) * elempack;

            __m256i _v2 = _mm256_set1_epi16(2);
            __m256i _v4 = _mm256_set1_epi16(4);
            __m256i _v5 = _mm256_set1_epi16(5);

            for (int m = 0; m < 6; m++)
            {
                __m256i _r0 = _mm256_setzero_si256();
                __m256i _r1 = _mm256_setzero_si256();
                __m256i _r2 = _mm256_setzero_si256();
                __m256i _r3 = _mm256_setzero_si256();
                __m256i _r4 = _mm256_setzero_si256();
                __m256i _r5 = _mm256_setzero_si256();

                if (ti * 4 + m < h)
                {
                    if (elempack == 16)
                    {
                        _r0 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)r0));
                        if (tj * 4 + 1 < w) _r1 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(r0 + 16)));
                        if (tj * 4 + 2 < w) _r2 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(r0 + 32)));
                        if (tj * 4 + 3 < w) _r3 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(r0 + 48)));
                        if (tj * 4 + 4 < w) _r4 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(r0 + 64)));
                        if (tj * 4 + 5 < w) _r5 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(r0 + 80)));
                    }
                    if (elempack == 8)
                    {
                        const signed char* r1 = r0 + N;

                        _r0 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)r0), _mm_loadl_epi64((const __m128i*)r1)));
                        if (tj * 4 + 1 < w) _r1 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r0 + 8)), _mm_loadl_epi64((const __m128i*)(r1 + 8))));
                        if (tj * 4 + 2 < w) _r2 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r0 + 16)), _mm_loadl_epi64((const __m128i*)(r1 + 16))));
                        if (tj * 4 + 3 < w) _r3 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r0 + 24)), _mm_loadl_epi64((const __m128i*)(r1 + 24))));
                        if (tj * 4 + 4 < w) _r4 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r0 + 32)), _mm_loadl_epi64((const __m128i*)(r1 + 32))));
                        if (tj * 4 + 5 < w) _r5 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r0 + 40)), _mm_loadl_epi64((const __m128i*)(r1 + 40))));
                    }
                    if (elempack == 1)
                    {
                        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                        _r0 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)r0, 1)));
                        if (tj * 4 + 1 < w) _r1 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0 + 1), 1)));
                        if (tj * 4 + 2 < w) _r2 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0 + 2), 1)));
                        if (tj * 4 + 3 < w) _r3 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0 + 3), 1)));
                        if (tj * 4 + 4 < w) _r4 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0 + 4), 1)));
                        if (tj * 4 + 5 < w) _r5 = _mm256_cvtepi8_epi16(_mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0 + 5), 1)));
                    }
                }

                __m256i _tmp12a = _mm256_sub_epi16(_r3, _mm256_mullo_epi16(_r1, _v4));
                __m256i _tmp12b = _mm256_sub_epi16(_r4, _mm256_mullo_epi16(_r2, _v4));
                __m256i _tmp34a = _mm256_mullo_epi16(_mm256_sub_epi16(_r3, _r1), _v2);
                __m256i _tmp34b = _mm256_sub_epi16(_r4, _r2);

                __m256i _tmp0 = _mm256_add_epi16(_r4, _mm256_sub_epi16(_mm256_mullo_epi16(_r0, _v4), _mm256_mullo_epi16(_r2, _v5)));
                __m256i _tmp1 = _mm256_add_epi16(_tmp12b, _tmp12a);
                __m256i _tmp2 = _mm256_sub_epi16(_tmp12b, _tmp12a);
                __m256i _tmp3 = _mm256_add_epi16(_tmp34b, _tmp34a);
                __m256i _tmp4 = _mm256_sub_epi16(_tmp34b, _tmp34a);
                __m256i _tmp5 = _mm256_add_epi16(_r5, _mm256_sub_epi16(_mm256_mullo_epi16(_r1, _v4), _mm256_mullo_epi16(_r3, _v5)));

                _mm256_store_si256((__m256i*)tmp[0][m], _tmp0);
                _mm256_store_si256((__m256i*)tmp[1][m], _tmp1);
                _mm256_store_si256((__m256i*)tmp[2][m], _tmp2);
                _mm256_store_si256((__m256i*)tmp[3][m], _tmp3);
                _mm256_store_si256((__m256i*)tmp[4][m], _tmp4);
                _mm256_store_si256((__m256i*)tmp[5][m], _tmp5);

                r0 += w * elempack;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj * 16;
            short* p1 = p0 + max_jj * 16;
            short* p2 = p0 + max_jj * 16 * 2;
            short* p3 = p0 + max_jj * 16 * 3;
            short* p4 = p0 + max_jj * 16 * 4;
            short* p5 = p0 + max_jj * 16 * 5;

            for (int m = 0; m < 6; m++)
            {
                __m256i _r0 = _mm256_load_si256((const __m256i*)tmp[m][0]);
                __m256i _r1 = _mm256_load_si256((const __m256i*)tmp[m][1]);
                __m256i _r2 = _mm256_load_si256((const __m256i*)tmp[m][2]);
                __m256i _r3 = _mm256_load_si256((const __m256i*)tmp[m][3]);
                __m256i _r4 = _mm256_load_si256((const __m256i*)tmp[m][4]);
                __m256i _r5 = _mm256_load_si256((const __m256i*)tmp[m][5]);

                __m256i _tmp12a = _mm256_sub_epi16(_r3, _mm256_mullo_epi16(_r1, _v4));
                __m256i _tmp12b = _mm256_sub_epi16(_r4, _mm256_mullo_epi16(_r2, _v4));
                __m256i _tmp34a = _mm256_mullo_epi16(_mm256_sub_epi16(_r3, _r1), _v2);
                __m256i _tmp34b = _mm256_sub_epi16(_r4, _r2);

                __m256i _tmp0 = _mm256_add_epi16(_r4, _mm256_sub_epi16(_mm256_mullo_epi16(_r0, _v4), _mm256_mullo_epi16(_r2, _v5)));
                __m256i _tmp1 = _mm256_add_epi16(_tmp12b, _tmp12a);
                __m256i _tmp2 = _mm256_sub_epi16(_tmp12b, _tmp12a);
                __m256i _tmp3 = _mm256_add_epi16(_tmp34b, _tmp34a);
                __m256i _tmp4 = _mm256_sub_epi16(_tmp34b, _tmp34a);
                __m256i _tmp5 = _mm256_add_epi16(_r5, _mm256_sub_epi16(_mm256_mullo_epi16(_r1, _v4), _mm256_mullo_epi16(_r3, _v5)));

                _mm256_store_si256((__m256i*)p0, _tmp0);
                _mm256_store_si256((__m256i*)p1, _tmp1);
                _mm256_store_si256((__m256i*)p2, _tmp2);
                _mm256_store_si256((__m256i*)p3, _tmp3);
                _mm256_store_si256((__m256i*)p4, _tmp4);
                _mm256_store_si256((__m256i*)p5, _tmp5);

                p0 += max_jj * 6 * 16;
                p1 += max_jj * 6 * 16;
                p2 += max_jj * 6 * 16;
                p3 += max_jj * 6 * 16;
                p4 += max_jj * 6 * 16;
                p5 += max_jj * 6 * 16;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 16;
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
#else // __AVX512F__
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
#endif // __AVX512F__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

#ifdef _MSC_VER
        __declspec(align(32))
#else
        __attribute__((aligned(32)))
#endif
        short tmp[6][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel((k + kk) / elempack).row<const signed char>(ti * 4) + (tj * 4) * elempack;

            __m128i _v2 = _mm_set1_epi16(2);
            __m128i _v4 = _mm_set1_epi16(4);
            __m128i _v5 = _mm_set1_epi16(5);

            for (int m = 0; m < 6; m++)
            {
                __m128i _r0 = _mm_setzero_si128();
                __m128i _r1 = _mm_setzero_si128();
                __m128i _r2 = _mm_setzero_si128();
                __m128i _r3 = _mm_setzero_si128();
                __m128i _r4 = _mm_setzero_si128();
                __m128i _r5 = _mm_setzero_si128();

                if (ti * 4 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = _mm_loadl_epi64((const __m128i*)r0);
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
                        if (tj * 4 + 1 < w)
                        {
                            _r1 = _mm_loadl_epi64((const __m128i*)(r0 + 8));
                            _r1 = _mm_unpacklo_epi8(_r1, _mm_cmpgt_epi8(_mm_setzero_si128(), _r1));
                        }
                        if (tj * 4 + 2 < w)
                        {
                            _r2 = _mm_loadl_epi64((const __m128i*)(r0 + 16));
                            _r2 = _mm_unpacklo_epi8(_r2, _mm_cmpgt_epi8(_mm_setzero_si128(), _r2));
                        }
                        if (tj * 4 + 3 < w)
                        {
                            _r3 = _mm_loadl_epi64((const __m128i*)(r0 + 24));
                            _r3 = _mm_unpacklo_epi8(_r3, _mm_cmpgt_epi8(_mm_setzero_si128(), _r3));
                        }
                        if (tj * 4 + 4 < w)
                        {
                            _r4 = _mm_loadl_epi64((const __m128i*)(r0 + 32));
                            _r4 = _mm_unpacklo_epi8(_r4, _mm_cmpgt_epi8(_mm_setzero_si128(), _r4));
                        }
                        if (tj * 4 + 5 < w)
                        {
                            _r5 = _mm_loadl_epi64((const __m128i*)(r0 + 40));
                            _r5 = _mm_unpacklo_epi8(_r5, _mm_cmpgt_epi8(_mm_setzero_si128(), _r5));
                        }
                    }
                    if (elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                        _r0 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0, _vindex, 1)));
                        if (tj * 4 + 1 < w) _r1 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(r0 + 1), _vindex, 1)));
                        if (tj * 4 + 2 < w) _r2 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(r0 + 2), _vindex, 1)));
                        if (tj * 4 + 3 < w) _r3 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(r0 + 3), _vindex, 1)));
                        if (tj * 4 + 4 < w) _r4 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(r0 + 4), _vindex, 1)));
                        if (tj * 4 + 5 < w) _r5 = _mm_cvtepi8_epi16(_mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(r0 + 5), _vindex, 1)));
#else
                        __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                        __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                        __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0, _vindex, 1), _sindex88);
                        _r0 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1)));
                        if (tj * 4 + 1 < w)
                        {
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(r0 + 1), _vindex, 1), _sindex88);
                            _r1 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1)));
                        }
                        if (tj * 4 + 2 < w)
                        {
                            __m256i _val2_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(r0 + 2), _vindex, 1), _sindex88);
                            _r2 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val2_32, 0), _mm256_extracti128_si256(_val2_32, 1)));
                        }
                        if (tj * 4 + 3 < w)
                        {
                            __m256i _val3_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(r0 + 3), _vindex, 1), _sindex88);
                            _r3 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val3_32, 0), _mm256_extracti128_si256(_val3_32, 1)));
                        }
                        if (tj * 4 + 4 < w)
                        {
                            __m256i _val4_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(r0 + 4), _vindex, 1), _sindex88);
                            _r4 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val4_32, 0), _mm256_extracti128_si256(_val4_32, 1)));
                        }
                        if (tj * 4 + 5 < w)
                        {
                            __m256i _val5_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(r0 + 5), _vindex, 1), _sindex88);
                            _r5 = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(_mm256_extracti128_si256(_val5_32, 0), _mm256_extracti128_si256(_val5_32, 1)));
                        }
#endif // __AVX512F__
#else  // __AVX2__
                        const signed char* r1 = r0 + N;
                        const signed char* r2 = r0 + N * 2;
                        const signed char* r3 = r0 + N * 3;
                        const signed char* r4 = r0 + N * 4;
                        const signed char* r5 = r0 + N * 5;
                        const signed char* r6 = r0 + N * 6;
                        const signed char* r7 = r0 + N * 7;

                        __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0);
                        __m128i _t1 = _mm_loadl_epi64((const __m128i*)r1);
                        __m128i _t2 = _mm_loadl_epi64((const __m128i*)r2);
                        __m128i _t3 = _mm_loadl_epi64((const __m128i*)r3);
                        __m128i _t4 = _mm_loadl_epi64((const __m128i*)r4);
                        __m128i _t5 = _mm_loadl_epi64((const __m128i*)r5);
                        __m128i _t6 = _mm_loadl_epi64((const __m128i*)r6);
                        __m128i _t7 = _mm_loadl_epi64((const __m128i*)r7);

                        __m128i _t01 = _mm_unpacklo_epi8(_t0, _t1);
                        __m128i _t23 = _mm_unpacklo_epi8(_t2, _t3);
                        __m128i _t45 = _mm_unpacklo_epi8(_t4, _t5);
                        __m128i _t67 = _mm_unpacklo_epi8(_t6, _t7);
                        _t0 = _mm_unpacklo_epi16(_t01, _t23);
                        _t1 = _mm_unpacklo_epi16(_t45, _t67);
                        _t2 = _mm_unpacklo_epi32(_t0, _t1);
                        _t3 = _mm_unpackhi_epi32(_t0, _t1);

                        __m128i _extt2 = _mm_cmpgt_epi8(_mm_setzero_si128(), _t2);
                        __m128i _extt3 = _mm_cmpgt_epi8(_mm_setzero_si128(), _t3);

                        _r0 = _mm_unpacklo_epi8(_t2, _extt2);
                        if (tj * 4 + 1 < w) _r1 = _mm_unpackhi_epi8(_t2, _extt2);
                        if (tj * 4 + 2 < w) _r2 = _mm_unpacklo_epi8(_t3, _extt3);
                        if (tj * 4 + 3 < w) _r3 = _mm_unpackhi_epi8(_t3, _extt3);
                        if (tj * 4 + 4 < w) _r4 = _mm_setr_epi16(r0[4], r1[4], r2[4], r3[4], r4[4], r5[4], r6[4], r7[4]);
                        if (tj * 4 + 5 < w) _r5 = _mm_setr_epi16(r0[5], r1[5], r2[5], r3[5], r4[5], r5[5], r6[5], r7[5]);
#endif // __AVX2__
                    }
                }

                __m128i _tmp12a = _mm_sub_epi16(_r3, _mm_mullo_epi16(_r1, _v4));
                __m128i _tmp12b = _mm_sub_epi16(_r4, _mm_mullo_epi16(_r2, _v4));
                __m128i _tmp34a = _mm_mullo_epi16(_mm_sub_epi16(_r3, _r1), _v2);
                __m128i _tmp34b = _mm_sub_epi16(_r4, _r2);

                __m128i _tmp0 = _mm_add_epi16(_r4, _mm_sub_epi16(_mm_mullo_epi16(_r0, _v4), _mm_mullo_epi16(_r2, _v5)));
                __m128i _tmp1 = _mm_add_epi16(_tmp12b, _tmp12a);
                __m128i _tmp2 = _mm_sub_epi16(_tmp12b, _tmp12a);
                __m128i _tmp3 = _mm_add_epi16(_tmp34b, _tmp34a);
                __m128i _tmp4 = _mm_sub_epi16(_tmp34b, _tmp34a);
                __m128i _tmp5 = _mm_add_epi16(_r5, _mm_sub_epi16(_mm_mullo_epi16(_r1, _v4), _mm_mullo_epi16(_r3, _v5)));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_storeu_si128((__m128i*)tmp[1][m], _tmp1);
                _mm_storeu_si128((__m128i*)tmp[2][m], _tmp2);
                _mm_storeu_si128((__m128i*)tmp[3][m], _tmp3);
                _mm_storeu_si128((__m128i*)tmp[4][m], _tmp4);
                _mm_storeu_si128((__m128i*)tmp[5][m], _tmp5);
#else
                _mm_store_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_store_si128((__m128i*)tmp[1][m], _tmp1);
                _mm_store_si128((__m128i*)tmp[2][m], _tmp2);
                _mm_store_si128((__m128i*)tmp[3][m], _tmp3);
                _mm_store_si128((__m128i*)tmp[4][m], _tmp4);
                _mm_store_si128((__m128i*)tmp[5][m], _tmp5);
#endif

                r0 += w * elempack;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj * 8;
            short* p1 = p0 + max_jj * 8;
            short* p2 = p0 + max_jj * 8 * 2;
            short* p3 = p0 + max_jj * 8 * 3;
            short* p4 = p0 + max_jj * 8 * 4;
            short* p5 = p0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128i _r0 = _mm_loadu_si128((const __m128i*)tmp[m][0]);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)tmp[m][1]);
                __m128i _r2 = _mm_loadu_si128((const __m128i*)tmp[m][2]);
                __m128i _r3 = _mm_loadu_si128((const __m128i*)tmp[m][3]);
                __m128i _r4 = _mm_loadu_si128((const __m128i*)tmp[m][4]);
                __m128i _r5 = _mm_loadu_si128((const __m128i*)tmp[m][5]);
#else
                __m128i _r0 = _mm_load_si128((const __m128i*)tmp[m][0]);
                __m128i _r1 = _mm_load_si128((const __m128i*)tmp[m][1]);
                __m128i _r2 = _mm_load_si128((const __m128i*)tmp[m][2]);
                __m128i _r3 = _mm_load_si128((const __m128i*)tmp[m][3]);
                __m128i _r4 = _mm_load_si128((const __m128i*)tmp[m][4]);
                __m128i _r5 = _mm_load_si128((const __m128i*)tmp[m][5]);
#endif

                __m128i _tmp12a = _mm_sub_epi16(_r3, _mm_mullo_epi16(_r1, _v4));
                __m128i _tmp12b = _mm_sub_epi16(_r4, _mm_mullo_epi16(_r2, _v4));
                __m128i _tmp34a = _mm_mullo_epi16(_mm_sub_epi16(_r3, _r1), _v2);
                __m128i _tmp34b = _mm_sub_epi16(_r4, _r2);

                __m128i _tmp0 = _mm_add_epi16(_r4, _mm_sub_epi16(_mm_mullo_epi16(_r0, _v4), _mm_mullo_epi16(_r2, _v5)));
                __m128i _tmp1 = _mm_add_epi16(_tmp12b, _tmp12a);
                __m128i _tmp2 = _mm_sub_epi16(_tmp12b, _tmp12a);
                __m128i _tmp3 = _mm_add_epi16(_tmp34b, _tmp34a);
                __m128i _tmp4 = _mm_sub_epi16(_tmp34b, _tmp34a);
                __m128i _tmp5 = _mm_add_epi16(_r5, _mm_sub_epi16(_mm_mullo_epi16(_r1, _v4), _mm_mullo_epi16(_r3, _v5)));

                _mm_store_si128((__m128i*)p0, _tmp0);
                _mm_store_si128((__m128i*)p1, _tmp1);
                _mm_store_si128((__m128i*)p2, _tmp2);
                _mm_store_si128((__m128i*)p3, _tmp3);
                _mm_store_si128((__m128i*)p4, _tmp4);
                _mm_store_si128((__m128i*)p5, _tmp5);

                p0 += max_jj * 6 * 8;
                p1 += max_jj * 6 * 8;
                p2 += max_jj * 6 * 8;
                p3 += max_jj * 6 * 8;
                p4 += max_jj * 6 * 8;
                p5 += max_jj * 6 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __SSE2__
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __SSE2__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        short tmp[6][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel(k + kk).row<const signed char>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                signed char r00 = 0;
                signed char r01 = 0;
                signed char r10 = 0;
                signed char r11 = 0;
                signed char r20 = 0;
                signed char r21 = 0;
                signed char r30 = 0;
                signed char r31 = 0;
                signed char r40 = 0;
                signed char r41 = 0;
                signed char r50 = 0;
                signed char r51 = 0;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 4 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 4 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 4 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                        if (tj * 4 + 4 < w)
                        {
                            r40 = r0[4];
                            r41 = r1[4];
                        }
                        if (tj * 4 + 5 < w)
                        {
                            r50 = r0[5];
                            r51 = r1[5];
                        }
                    }
                }

                short tmp120a = r30 - r10 * 4;
                short tmp121a = r31 - r11 * 4;
                short tmp120b = r40 - r20 * 4;
                short tmp121b = r41 - r21 * 4;
                short tmp340a = (r30 - r10) * 2;
                short tmp341a = (r31 - r11) * 2;
                short tmp340b = r40 - r20;
                short tmp341b = r41 - r21;

                tmp[0][m][0] = r40 + r00 * 4 - r20 * 5;
                tmp[0][m][1] = r41 + r01 * 4 - r21 * 5;
                tmp[1][m][0] = tmp120b + tmp120a;
                tmp[1][m][1] = tmp121b + tmp121a;
                tmp[2][m][0] = tmp120b - tmp120a;
                tmp[2][m][1] = tmp121b - tmp121a;
                tmp[3][m][0] = tmp340b + tmp340a;
                tmp[3][m][1] = tmp341b + tmp341a;
                tmp[4][m][0] = tmp340b - tmp340a;
                tmp[4][m][1] = tmp341b - tmp341a;
                tmp[5][m][0] = r50 + r10 * 4 - r30 * 5;
                tmp[5][m][1] = r51 + r11 * 4 - r31 * 5;

                r0 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj * 2;
            short* p1 = p0 + max_jj * 2;
            short* p2 = p0 + max_jj * 2 * 2;
            short* p3 = p0 + max_jj * 2 * 3;
            short* p4 = p0 + max_jj * 2 * 4;
            short* p5 = p0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                short r00 = tmp[m][0][0];
                short r01 = tmp[m][0][1];
                short r10 = tmp[m][1][0];
                short r11 = tmp[m][1][1];
                short r20 = tmp[m][2][0];
                short r21 = tmp[m][2][1];
                short r30 = tmp[m][3][0];
                short r31 = tmp[m][3][1];
                short r40 = tmp[m][4][0];
                short r41 = tmp[m][4][1];
                short r50 = tmp[m][5][0];
                short r51 = tmp[m][5][1];

                short tmp120a = r30 - r10 * 4;
                short tmp121a = r31 - r11 * 4;
                short tmp120b = r40 - r20 * 4;
                short tmp121b = r41 - r21 * 4;
                short tmp340a = (r30 - r10) * 2;
                short tmp341a = (r31 - r11) * 2;
                short tmp340b = r40 - r20;
                short tmp341b = r41 - r21;

                p0[0] = r40 + r00 * 4 - r20 * 5;
                p0[1] = r41 + r01 * 4 - r21 * 5;
                p1[0] = tmp120b + tmp120a;
                p1[1] = tmp121b + tmp121a;
                p2[0] = tmp120b - tmp120a;
                p2[1] = tmp121b - tmp121a;
                p3[0] = tmp340b + tmp340a;
                p3[1] = tmp341b + tmp341a;
                p4[0] = tmp340b - tmp340a;
                p4[1] = tmp341b - tmp341a;
                p5[0] = r50 + r10 * 4 - r30 * 5;
                p5[1] = r51 + r11 * 4 - r31 * 5;

                p0 += max_jj * 6 * 2;
                p1 += max_jj * 6 * 2;
                p2 += max_jj * 6 * 2;
                p3 += max_jj * 6 * 2;
                p4 += max_jj * 6 * 2;
                p5 += max_jj * 6 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        short tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0123 = bottom_blob.channel(k + kk).row<const signed char>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                signed char r0 = 0;
                signed char r1 = 0;
                signed char r2 = 0;
                signed char r3 = 0;
                signed char r4 = 0;
                signed char r5 = 0;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 4 + 1 < w) r1 = r0123[1];
                        if (tj * 4 + 2 < w) r2 = r0123[2];
                        if (tj * 4 + 3 < w) r3 = r0123[3];
                        if (tj * 4 + 4 < w) r4 = r0123[4];
                        if (tj * 4 + 5 < w) r5 = r0123[5];
                    }
                }

                short tmp12a = r3 - r1 * 4;
                short tmp12b = r4 - r2 * 4;
                short tmp34a = (r3 - r1) * 2;
                short tmp34b = r4 - r2;

                tmp[0][m] = r4 + r0 * 4 - r2 * 5;
                tmp[1][m] = tmp12b + tmp12a;
                tmp[2][m] = tmp12b - tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r5 + r1 * 4 - r3 * 5;

                r0123 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj;
            short* p1 = p0 + max_jj;
            short* p2 = p0 + max_jj * 2;
            short* p3 = p0 + max_jj * 3;
            short* p4 = p0 + max_jj * 4;
            short* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];
                short r3 = tmp[m][3];
                short r4 = tmp[m][4];
                short r5 = tmp[m][5];

                short tmp12a = r3 - r1 * 4;
                short tmp12b = r4 - r2 * 4;
                short tmp34a = (r3 - r1) * 2;
                short tmp34b = r4 - r2;

                p0[0] = r4 + r0 * 4 - r2 * 5;
                p1[0] = tmp12b + tmp12a;
                p2[0] = tmp12b - tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r5 + r1 * 4 - r3 * 5;

                p0 += max_jj * 6;
                p1 += max_jj * 6;
                p2 += max_jj * 6;
                p3 += max_jj * 6;
                p4 += max_jj * 6;
                p5 += max_jj * 6;
            }
        }
    }
}

static inline void conv3x3s1_winograd43_transform_output_tile_int8(const Mat& top_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    // const int otm[4][6] = {
    //     {1, 1,  1, 1,  1, 0},
    //     {0, 1, -1, 2, -2, 0},
    //     {0, 1,  1, 4,  4, 0},
    //     {0, 1, -1, 8, -8, 1}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 3) / 4;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        int tmp[4][6][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj * 16;
            const int* r1 = r0 + max_jj * 16;
            const int* r2 = r0 + max_jj * 16 * 2;
            const int* r3 = r0 + max_jj * 16 * 3;
            const int* r4 = r0 + max_jj * 16 * 4;
            const int* r5 = r0 + max_jj * 16 * 5;

            for (int m = 0; m < 5; m++)
            {
                __m512i _r0 = _mm512_load_si512((const __m512i*)r0);
                __m512i _r1 = _mm512_load_si512((const __m512i*)r1);
                __m512i _r2 = _mm512_load_si512((const __m512i*)r2);
                __m512i _r3 = _mm512_load_si512((const __m512i*)r3);
                __m512i _r4 = _mm512_load_si512((const __m512i*)r4);
                __m512i _r5 = _mm512_load_si512((const __m512i*)r5);

                __m512i _tmp02a = _mm512_add_epi32(_r1, _r2);
                __m512i _tmp02b = _mm512_add_epi32(_r3, _r4);
                __m512i _tmp13a = _mm512_sub_epi32(_r1, _r2);
                __m512i _tmp13b = _mm512_sub_epi32(_r3, _r4);

                __m512i _tmp0 = _mm512_add_epi32(_mm512_add_epi32(_tmp02a, _tmp02b), _r0);
                __m512i _tmp1 = _mm512_add_epi32(_tmp13a, _mm512_slli_epi32(_tmp13b, 1));
                __m512i _tmp2 = _mm512_add_epi32(_tmp02a, _mm512_slli_epi32(_tmp02b, 2));
                __m512i _tmp3 = _mm512_add_epi32(_mm512_add_epi32(_tmp13a, _mm512_slli_epi32(_tmp13b, 3)), _mm512_slli_epi32(_r5, 2));

                _mm512_store_si512((__m512i*)tmp[0][m], _tmp0);
                _mm512_store_si512((__m512i*)tmp[1][m], _tmp1);
                _mm512_store_si512((__m512i*)tmp[2][m], _tmp2);
                _mm512_store_si512((__m512i*)tmp[3][m], _tmp3);

                r0 += max_jj * 6 * 16;
                r1 += max_jj * 6 * 16;
                r2 += max_jj * 6 * 16;
                r3 += max_jj * 6 * 16;
                r4 += max_jj * 6 * 16;
                r5 += max_jj * 6 * 16;
            }
            for (int m = 5; m < 6; m++)
            {
                __m512i _r0 = _mm512_load_si512((const __m512i*)r0);
                __m512i _r1 = _mm512_load_si512((const __m512i*)r1);
                __m512i _r2 = _mm512_load_si512((const __m512i*)r2);
                __m512i _r3 = _mm512_load_si512((const __m512i*)r3);
                __m512i _r4 = _mm512_load_si512((const __m512i*)r4);
                __m512i _r5 = _mm512_load_si512((const __m512i*)r5);

                __m512i _tmp02a = _mm512_add_epi32(_r1, _r2);
                __m512i _tmp02b = _mm512_add_epi32(_r3, _r4);
                __m512i _tmp13a = _mm512_sub_epi32(_r1, _r2);
                __m512i _tmp13b = _mm512_sub_epi32(_r3, _r4);

                __m512i _tmp0 = _mm512_add_epi32(_mm512_add_epi32(_tmp02a, _tmp02b), _r0);
                __m512i _tmp1 = _mm512_add_epi32(_tmp13a, _mm512_slli_epi32(_tmp13b, 1));
                __m512i _tmp2 = _mm512_add_epi32(_tmp02a, _mm512_slli_epi32(_tmp02b, 2));
                __m512i _tmp3 = _mm512_add_epi32(_mm512_add_epi32(_tmp13a, _mm512_slli_epi32(_tmp13b, 3)), _mm512_slli_epi32(_r5, 2));

                _tmp0 = _mm512_slli_epi32(_tmp0, 2);
                _tmp1 = _mm512_slli_epi32(_tmp1, 2);
                _tmp2 = _mm512_slli_epi32(_tmp2, 2);
                _tmp3 = _mm512_slli_epi32(_tmp3, 2);

                _mm512_store_si512((__m512i*)tmp[0][m], _tmp0);
                _mm512_store_si512((__m512i*)tmp[1][m], _tmp1);
                _mm512_store_si512((__m512i*)tmp[2][m], _tmp2);
                _mm512_store_si512((__m512i*)tmp[3][m], _tmp3);

                r0 += max_jj * 6 * 16;
                r1 += max_jj * 6 * 16;
                r2 += max_jj * 6 * 16;
                r3 += max_jj * 6 * 16;
                r4 += max_jj * 6 * 16;
                r5 += max_jj * 6 * 16;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __m512i _r0 = _mm512_load_si512((const __m512i*)tmp[m][0]);
                __m512i _r1 = _mm512_load_si512((const __m512i*)tmp[m][1]);
                __m512i _r2 = _mm512_load_si512((const __m512i*)tmp[m][2]);
                __m512i _r3 = _mm512_load_si512((const __m512i*)tmp[m][3]);
                __m512i _r4 = _mm512_load_si512((const __m512i*)tmp[m][4]);
                __m512i _r5 = _mm512_load_si512((const __m512i*)tmp[m][5]);

                __m512i _tmp02a = _mm512_add_epi32(_r1, _r2);
                __m512i _tmp02b = _mm512_add_epi32(_r3, _r4);
                __m512i _tmp13a = _mm512_sub_epi32(_r1, _r2);
                __m512i _tmp13b = _mm512_sub_epi32(_r3, _r4);

                __m512i _tmp0 = _mm512_add_epi32(_mm512_add_epi32(_tmp02a, _tmp02b), _r0);
                __m512i _tmp1 = _mm512_add_epi32(_tmp13a, _mm512_slli_epi32(_tmp13b, 1));
                __m512i _tmp2 = _mm512_add_epi32(_tmp02a, _mm512_slli_epi32(_tmp02b, 2));
                __m512i _tmp3 = _mm512_add_epi32(_mm512_add_epi32(_tmp13a, _mm512_slli_epi32(_tmp13b, 3)), _r5);

                // TODO use integer trick for division by 576
                __m512 _v576 = _mm512_set1_ps(1.0 / 576);
                _tmp0 = _mm512_cvttps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_tmp0), _v576));
                _tmp1 = _mm512_cvttps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_tmp1), _v576));
                _tmp2 = _mm512_cvttps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_tmp2), _v576));
                _tmp3 = _mm512_cvttps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_tmp3), _v576));

                if (out_elempack == 16)
                {
                    _mm512_store_si512((__m512i*)outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm512_store_si512((__m512i*)(outptr0 + 16), _tmp1);
                    if (tj * 4 + 2 < outw) _mm512_store_si512((__m512i*)(outptr0 + 32), _tmp2);
                    if (tj * 4 + 3 < outw) _mm512_store_si512((__m512i*)(outptr0 + 48), _tmp3);
                }
                if (out_elempack == 8)
                {
                    int* outptr1 = outptr0 + N;

                    _mm256_store_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_tmp0, 0));
                    _mm256_store_si256((__m256i*)outptr1, _mm512_extracti32x8_epi32(_tmp0, 1));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm256_store_si256((__m256i*)(outptr0 + 8), _mm512_extracti32x8_epi32(_tmp1, 0));
                        _mm256_store_si256((__m256i*)(outptr1 + 8), _mm512_extracti32x8_epi32(_tmp1, 1));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm256_store_si256((__m256i*)(outptr0 + 16), _mm512_extracti32x8_epi32(_tmp2, 0));
                        _mm256_store_si256((__m256i*)(outptr1 + 16), _mm512_extracti32x8_epi32(_tmp2, 1));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm256_store_si256((__m256i*)(outptr0 + 24), _mm512_extracti32x8_epi32(_tmp3, 0));
                        _mm256_store_si256((__m256i*)(outptr1 + 24), _mm512_extracti32x8_epi32(_tmp3, 1));
                    }
                }
                if (out_elempack == 4)
                {
                    int* outptr1 = outptr0 + N;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;

                    _mm_store_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_tmp0, 0));
                    _mm_store_si128((__m128i*)outptr1, _mm512_extracti32x4_epi32(_tmp0, 1));
                    _mm_store_si128((__m128i*)outptr2, _mm512_extracti32x4_epi32(_tmp0, 2));
                    _mm_store_si128((__m128i*)outptr3, _mm512_extracti32x4_epi32(_tmp0, 3));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm_store_si128((__m128i*)(outptr0 + 4), _mm512_extracti32x4_epi32(_tmp1, 0));
                        _mm_store_si128((__m128i*)(outptr1 + 4), _mm512_extracti32x4_epi32(_tmp1, 1));
                        _mm_store_si128((__m128i*)(outptr2 + 4), _mm512_extracti32x4_epi32(_tmp1, 2));
                        _mm_store_si128((__m128i*)(outptr3 + 4), _mm512_extracti32x4_epi32(_tmp1, 3));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm_store_si128((__m128i*)(outptr0 + 8), _mm512_extracti32x4_epi32(_tmp2, 0));
                        _mm_store_si128((__m128i*)(outptr1 + 8), _mm512_extracti32x4_epi32(_tmp2, 1));
                        _mm_store_si128((__m128i*)(outptr2 + 8), _mm512_extracti32x4_epi32(_tmp2, 2));
                        _mm_store_si128((__m128i*)(outptr3 + 8), _mm512_extracti32x4_epi32(_tmp2, 3));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm_store_si128((__m128i*)(outptr0 + 12), _mm512_extracti32x4_epi32(_tmp3, 0));
                        _mm_store_si128((__m128i*)(outptr1 + 12), _mm512_extracti32x4_epi32(_tmp3, 1));
                        _mm_store_si128((__m128i*)(outptr2 + 12), _mm512_extracti32x4_epi32(_tmp3, 2));
                        _mm_store_si128((__m128i*)(outptr3 + 12), _mm512_extracti32x4_epi32(_tmp3, 3));
                    }
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                    _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                    _mm512_i32scatter_epi32(outptr0, _vindex, _tmp0, sizeof(int));
                    if (tj * 4 + 1 < outw) _mm512_i32scatter_epi32(outptr0 + 1, _vindex, _tmp1, sizeof(int));
                    if (tj * 4 + 2 < outw) _mm512_i32scatter_epi32(outptr0 + 2, _vindex, _tmp2, sizeof(int));
                    if (tj * 4 + 3 < outw) _mm512_i32scatter_epi32(outptr0 + 3, _vindex, _tmp3, sizeof(int));
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
#ifdef _MSC_VER
        __declspec(align(32))
#else
        __attribute__((aligned(32)))
#endif
        int tmp[4][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj * 8;
            const int* r1 = r0 + max_jj * 8;
            const int* r2 = r0 + max_jj * 8 * 2;
            const int* r3 = r0 + max_jj * 8 * 3;
            const int* r4 = r0 + max_jj * 8 * 4;
            const int* r5 = r0 + max_jj * 8 * 5;

            for (int m = 0; m < 5; m++)
            {
                __m256i _r0 = _mm256_load_si256((const __m256i*)r0);
                __m256i _r1 = _mm256_load_si256((const __m256i*)r1);
                __m256i _r2 = _mm256_load_si256((const __m256i*)r2);
                __m256i _r3 = _mm256_load_si256((const __m256i*)r3);
                __m256i _r4 = _mm256_load_si256((const __m256i*)r4);
                __m256i _r5 = _mm256_load_si256((const __m256i*)r5);

                __m256i _tmp02a = _mm256_add_epi32(_r1, _r2);
                __m256i _tmp02b = _mm256_add_epi32(_r3, _r4);
                __m256i _tmp13a = _mm256_sub_epi32(_r1, _r2);
                __m256i _tmp13b = _mm256_sub_epi32(_r3, _r4);

                __m256i _tmp0 = _mm256_add_epi32(_mm256_add_epi32(_tmp02a, _tmp02b), _r0);
                __m256i _tmp1 = _mm256_add_epi32(_tmp13a, _mm256_slli_epi32(_tmp13b, 1));
                __m256i _tmp2 = _mm256_add_epi32(_tmp02a, _mm256_slli_epi32(_tmp02b, 2));
                __m256i _tmp3 = _mm256_add_epi32(_mm256_add_epi32(_tmp13a, _mm256_slli_epi32(_tmp13b, 3)), _mm256_slli_epi32(_r5, 2));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm256_storeu_si256((__m256i*)tmp[0][m], _tmp0);
                _mm256_storeu_si256((__m256i*)tmp[1][m], _tmp1);
                _mm256_storeu_si256((__m256i*)tmp[2][m], _tmp2);
                _mm256_storeu_si256((__m256i*)tmp[3][m], _tmp3);
#else
                _mm256_store_si256((__m256i*)tmp[0][m], _tmp0);
                _mm256_store_si256((__m256i*)tmp[1][m], _tmp1);
                _mm256_store_si256((__m256i*)tmp[2][m], _tmp2);
                _mm256_store_si256((__m256i*)tmp[3][m], _tmp3);
#endif

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }
            for (int m = 5; m < 6; m++)
            {
                __m256i _r0 = _mm256_load_si256((const __m256i*)r0);
                __m256i _r1 = _mm256_load_si256((const __m256i*)r1);
                __m256i _r2 = _mm256_load_si256((const __m256i*)r2);
                __m256i _r3 = _mm256_load_si256((const __m256i*)r3);
                __m256i _r4 = _mm256_load_si256((const __m256i*)r4);
                __m256i _r5 = _mm256_load_si256((const __m256i*)r5);

                __m256i _tmp02a = _mm256_add_epi32(_r1, _r2);
                __m256i _tmp02b = _mm256_add_epi32(_r3, _r4);
                __m256i _tmp13a = _mm256_sub_epi32(_r1, _r2);
                __m256i _tmp13b = _mm256_sub_epi32(_r3, _r4);

                __m256i _tmp0 = _mm256_add_epi32(_mm256_add_epi32(_tmp02a, _tmp02b), _r0);
                __m256i _tmp1 = _mm256_add_epi32(_tmp13a, _mm256_slli_epi32(_tmp13b, 1));
                __m256i _tmp2 = _mm256_add_epi32(_tmp02a, _mm256_slli_epi32(_tmp02b, 2));
                __m256i _tmp3 = _mm256_add_epi32(_mm256_add_epi32(_tmp13a, _mm256_slli_epi32(_tmp13b, 3)), _mm256_slli_epi32(_r5, 2));

                _tmp0 = _mm256_slli_epi32(_tmp0, 2);
                _tmp1 = _mm256_slli_epi32(_tmp1, 2);
                _tmp2 = _mm256_slli_epi32(_tmp2, 2);
                _tmp3 = _mm256_slli_epi32(_tmp3, 2);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm256_storeu_si256((__m256i*)tmp[0][m], _tmp0);
                _mm256_storeu_si256((__m256i*)tmp[1][m], _tmp1);
                _mm256_storeu_si256((__m256i*)tmp[2][m], _tmp2);
                _mm256_storeu_si256((__m256i*)tmp[3][m], _tmp3);
#else
                _mm256_store_si256((__m256i*)tmp[0][m], _tmp0);
                _mm256_store_si256((__m256i*)tmp[1][m], _tmp1);
                _mm256_store_si256((__m256i*)tmp[2][m], _tmp2);
                _mm256_store_si256((__m256i*)tmp[3][m], _tmp3);
#endif

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)tmp[m][0]);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)tmp[m][1]);
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)tmp[m][2]);
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)tmp[m][3]);
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)tmp[m][4]);
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)tmp[m][5]);
#else
                __m256i _r0 = _mm256_load_si256((const __m256i*)tmp[m][0]);
                __m256i _r1 = _mm256_load_si256((const __m256i*)tmp[m][1]);
                __m256i _r2 = _mm256_load_si256((const __m256i*)tmp[m][2]);
                __m256i _r3 = _mm256_load_si256((const __m256i*)tmp[m][3]);
                __m256i _r4 = _mm256_load_si256((const __m256i*)tmp[m][4]);
                __m256i _r5 = _mm256_load_si256((const __m256i*)tmp[m][5]);
#endif

                __m256i _tmp02a = _mm256_add_epi32(_r1, _r2);
                __m256i _tmp02b = _mm256_add_epi32(_r3, _r4);
                __m256i _tmp13a = _mm256_sub_epi32(_r1, _r2);
                __m256i _tmp13b = _mm256_sub_epi32(_r3, _r4);

                __m256i _tmp0 = _mm256_add_epi32(_mm256_add_epi32(_tmp02a, _tmp02b), _r0);
                __m256i _tmp1 = _mm256_add_epi32(_tmp13a, _mm256_slli_epi32(_tmp13b, 1));
                __m256i _tmp2 = _mm256_add_epi32(_tmp02a, _mm256_slli_epi32(_tmp02b, 2));
                __m256i _tmp3 = _mm256_add_epi32(_mm256_add_epi32(_tmp13a, _mm256_slli_epi32(_tmp13b, 3)), _r5);

                // TODO use integer trick for division by 576
                __m256 _v576 = _mm256_set1_ps(1.0 / 576);
                _tmp0 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_tmp0), _v576));
                _tmp1 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_tmp1), _v576));
                _tmp2 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_tmp2), _v576));
                _tmp3 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_tmp3), _v576));

                if (out_elempack == 8)
                {
                    _mm256_store_si256((__m256i*)outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm256_store_si256((__m256i*)(outptr0 + 8), _tmp1);
                    if (tj * 4 + 2 < outw) _mm256_store_si256((__m256i*)(outptr0 + 16), _tmp2);
                    if (tj * 4 + 3 < outw) _mm256_store_si256((__m256i*)(outptr0 + 24), _tmp3);
                }
                if (out_elempack == 4)
                {
                    int* outptr1 = outptr0 + N;

                    _mm_store_si128((__m128i*)(outptr0), _mm256_extracti128_si256(_tmp0, 0));
                    _mm_store_si128((__m128i*)(outptr1), _mm256_extracti128_si256(_tmp0, 1));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm_store_si128((__m128i*)(outptr0 + 4), _mm256_extracti128_si256(_tmp1, 0));
                        _mm_store_si128((__m128i*)(outptr1 + 4), _mm256_extracti128_si256(_tmp1, 1));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm_store_si128((__m128i*)(outptr0 + 8), _mm256_extracti128_si256(_tmp2, 0));
                        _mm_store_si128((__m128i*)(outptr1 + 8), _mm256_extracti128_si256(_tmp2, 1));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm_store_si128((__m128i*)(outptr0 + 12), _mm256_extracti128_si256(_tmp3, 0));
                        _mm_store_si128((__m128i*)(outptr1 + 12), _mm256_extracti128_si256(_tmp3, 1));
                    }
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
                    _mm256_i32scatter_epi32(outptr0, _vindex, _tmp0, sizeof(int));
                    if (tj * 4 + 1 < outw) _mm256_i32scatter_epi32(outptr0 + 1, _vindex, _tmp1, sizeof(int));
                    if (tj * 4 + 2 < outw) _mm256_i32scatter_epi32(outptr0 + 2, _vindex, _tmp2, sizeof(int));
                    if (tj * 4 + 3 < outw) _mm256_i32scatter_epi32(outptr0 + 3, _vindex, _tmp3, sizeof(int));
#else
                    int tmp0[8];
                    int tmp1[8];
                    int tmp2[8];
                    int tmp3[8];
                    _mm256_storeu_si256((__m256i*)tmp0, _tmp0);
                    _mm256_storeu_si256((__m256i*)tmp1, _tmp1);
                    _mm256_storeu_si256((__m256i*)tmp2, _tmp2);
                    _mm256_storeu_si256((__m256i*)tmp3, _tmp3);

                    int* outptr1 = outptr0 + N;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;
                    int* outptr4 = outptr0 + N * 4;
                    int* outptr5 = outptr0 + N * 5;
                    int* outptr6 = outptr0 + N * 6;
                    int* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                        outptr4[2] = tmp2[4];
                        outptr5[2] = tmp2[5];
                        outptr6[2] = tmp2[6];
                        outptr7[2] = tmp2[7];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                        outptr4[3] = tmp3[4];
                        outptr5[3] = tmp3[5];
                        outptr6[3] = tmp3[6];
                        outptr7[3] = tmp3[7];
                    }
#endif // __AVX512F__
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        int tmp[4][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj * 4;
            const int* r1 = r0 + max_jj * 4;
            const int* r2 = r0 + max_jj * 4 * 2;
            const int* r3 = r0 + max_jj * 4 * 3;
            const int* r4 = r0 + max_jj * 4 * 4;
            const int* r5 = r0 + max_jj * 4 * 5;

            for (int m = 0; m < 5; m++)
            {
                __m128i _r0 = _mm_load_si128((const __m128i*)r0);
                __m128i _r1 = _mm_load_si128((const __m128i*)r1);
                __m128i _r2 = _mm_load_si128((const __m128i*)r2);
                __m128i _r3 = _mm_load_si128((const __m128i*)r3);
                __m128i _r4 = _mm_load_si128((const __m128i*)r4);
                __m128i _r5 = _mm_load_si128((const __m128i*)r5);

                __m128i _tmp02a = _mm_add_epi32(_r1, _r2);
                __m128i _tmp02b = _mm_add_epi32(_r3, _r4);
                __m128i _tmp13a = _mm_sub_epi32(_r1, _r2);
                __m128i _tmp13b = _mm_sub_epi32(_r3, _r4);

                __m128i _tmp0 = _mm_add_epi32(_mm_add_epi32(_tmp02a, _tmp02b), _r0);
                __m128i _tmp1 = _mm_add_epi32(_tmp13a, _mm_slli_epi32(_tmp13b, 1));
                __m128i _tmp2 = _mm_add_epi32(_tmp02a, _mm_slli_epi32(_tmp02b, 2));
                __m128i _tmp3 = _mm_add_epi32(_mm_add_epi32(_tmp13a, _mm_slli_epi32(_tmp13b, 3)), _mm_slli_epi32(_r5, 2));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_storeu_si128((__m128i*)tmp[1][m], _tmp1);
                _mm_storeu_si128((__m128i*)tmp[2][m], _tmp2);
                _mm_storeu_si128((__m128i*)tmp[3][m], _tmp3);
#else
                _mm_store_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_store_si128((__m128i*)tmp[1][m], _tmp1);
                _mm_store_si128((__m128i*)tmp[2][m], _tmp2);
                _mm_store_si128((__m128i*)tmp[3][m], _tmp3);
#endif

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }
            for (int m = 5; m < 6; m++)
            {
                __m128i _r0 = _mm_load_si128((const __m128i*)r0);
                __m128i _r1 = _mm_load_si128((const __m128i*)r1);
                __m128i _r2 = _mm_load_si128((const __m128i*)r2);
                __m128i _r3 = _mm_load_si128((const __m128i*)r3);
                __m128i _r4 = _mm_load_si128((const __m128i*)r4);
                __m128i _r5 = _mm_load_si128((const __m128i*)r5);

                __m128i _tmp02a = _mm_add_epi32(_r1, _r2);
                __m128i _tmp02b = _mm_add_epi32(_r3, _r4);
                __m128i _tmp13a = _mm_sub_epi32(_r1, _r2);
                __m128i _tmp13b = _mm_sub_epi32(_r3, _r4);

                __m128i _tmp0 = _mm_add_epi32(_mm_add_epi32(_tmp02a, _tmp02b), _r0);
                __m128i _tmp1 = _mm_add_epi32(_tmp13a, _mm_slli_epi32(_tmp13b, 1));
                __m128i _tmp2 = _mm_add_epi32(_tmp02a, _mm_slli_epi32(_tmp02b, 2));
                __m128i _tmp3 = _mm_add_epi32(_mm_add_epi32(_tmp13a, _mm_slli_epi32(_tmp13b, 3)), _mm_slli_epi32(_r5, 2));

                _tmp0 = _mm_slli_epi32(_tmp0, 2);
                _tmp1 = _mm_slli_epi32(_tmp1, 2);
                _tmp2 = _mm_slli_epi32(_tmp2, 2);
                _tmp3 = _mm_slli_epi32(_tmp3, 2);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_storeu_si128((__m128i*)tmp[1][m], _tmp1);
                _mm_storeu_si128((__m128i*)tmp[2][m], _tmp2);
                _mm_storeu_si128((__m128i*)tmp[3][m], _tmp3);
#else
                _mm_store_si128((__m128i*)tmp[0][m], _tmp0);
                _mm_store_si128((__m128i*)tmp[1][m], _tmp1);
                _mm_store_si128((__m128i*)tmp[2][m], _tmp2);
                _mm_store_si128((__m128i*)tmp[3][m], _tmp3);
#endif

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128i _r0 = _mm_loadu_si128((const __m128i*)tmp[m][0]);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)tmp[m][1]);
                __m128i _r2 = _mm_loadu_si128((const __m128i*)tmp[m][2]);
                __m128i _r3 = _mm_loadu_si128((const __m128i*)tmp[m][3]);
                __m128i _r4 = _mm_loadu_si128((const __m128i*)tmp[m][4]);
                __m128i _r5 = _mm_loadu_si128((const __m128i*)tmp[m][5]);
#else
                __m128i _r0 = _mm_load_si128((const __m128i*)tmp[m][0]);
                __m128i _r1 = _mm_load_si128((const __m128i*)tmp[m][1]);
                __m128i _r2 = _mm_load_si128((const __m128i*)tmp[m][2]);
                __m128i _r3 = _mm_load_si128((const __m128i*)tmp[m][3]);
                __m128i _r4 = _mm_load_si128((const __m128i*)tmp[m][4]);
                __m128i _r5 = _mm_load_si128((const __m128i*)tmp[m][5]);
#endif

                __m128i _tmp02a = _mm_add_epi32(_r1, _r2);
                __m128i _tmp02b = _mm_add_epi32(_r3, _r4);
                __m128i _tmp13a = _mm_sub_epi32(_r1, _r2);
                __m128i _tmp13b = _mm_sub_epi32(_r3, _r4);

                __m128i _tmp0 = _mm_add_epi32(_mm_add_epi32(_tmp02a, _tmp02b), _r0);
                __m128i _tmp1 = _mm_add_epi32(_tmp13a, _mm_slli_epi32(_tmp13b, 1));
                __m128i _tmp2 = _mm_add_epi32(_tmp02a, _mm_slli_epi32(_tmp02b, 2));
                __m128i _tmp3 = _mm_add_epi32(_mm_add_epi32(_tmp13a, _mm_slli_epi32(_tmp13b, 3)), _r5);

                // TODO use integer trick for division by 576
                __m128 _v576 = _mm_set1_ps(1.0 / 576);
                _tmp0 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_tmp0), _v576));
                _tmp1 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_tmp1), _v576));
                _tmp2 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_tmp2), _v576));
                _tmp3 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_tmp3), _v576));

                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm_store_si128((__m128i*)(outptr0 + 4), _tmp1);
                    if (tj * 4 + 2 < outw) _mm_store_si128((__m128i*)(outptr0 + 8), _tmp2);
                    if (tj * 4 + 3 < outw) _mm_store_si128((__m128i*)(outptr0 + 12), _tmp3);
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                    _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(N));
                    _mm_i32scatter_epi32(outptr0, _vindex, _tmp0, sizeof(int));
                    if (tj * 4 + 1 < outw) _mm_i32scatter_epi32(outptr0 + 1, _vindex, _tmp1, sizeof(int));
                    if (tj * 4 + 2 < outw) _mm_i32scatter_epi32(outptr0 + 2, _vindex, _tmp2, sizeof(int));
                    if (tj * 4 + 3 < outw) _mm_i32scatter_epi32(outptr0 + 3, _vindex, _tmp3, sizeof(int));
#else
                    int tmp0[4];
                    int tmp1[4];
                    int tmp2[4];
                    int tmp3[4];
                    _mm_storeu_si128((__m128i*)tmp0, _tmp0);
                    _mm_storeu_si128((__m128i*)tmp1, _tmp1);
                    _mm_storeu_si128((__m128i*)tmp2, _tmp2);
                    _mm_storeu_si128((__m128i*)tmp3, _tmp3);

                    int* outptr1 = outptr0 + N;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                    }
#endif // __AVX512F__
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        int tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj * 2;
            const int* r1 = r0 + max_jj * 2;
            const int* r2 = r0 + max_jj * 2 * 2;
            const int* r3 = r0 + max_jj * 2 * 3;
            const int* r4 = r0 + max_jj * 2 * 4;
            const int* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 5; m++)
            {
                int tmp02a0 = r1[0] + r2[0];
                int tmp02a1 = r1[1] + r2[1];
                int tmp02b0 = r3[0] + r4[0];
                int tmp02b1 = r3[1] + r4[1];
                int tmp13a0 = r1[0] - r2[0];
                int tmp13a1 = r1[1] - r2[1];
                int tmp13b0 = r3[0] - r4[0];
                int tmp13b1 = r3[1] - r4[1];

                int tmp00 = tmp02a0 + tmp02b0 + r0[0];
                int tmp01 = tmp02a1 + tmp02b1 + r0[1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + r5[0] * 4;
                int tmp31 = tmp13a1 + tmp13b1 * 8 + r5[1] * 4;

                tmp[0][m][0] = tmp00;
                tmp[0][m][1] = tmp01;
                tmp[1][m][0] = tmp10;
                tmp[1][m][1] = tmp11;
                tmp[2][m][0] = tmp20;
                tmp[2][m][1] = tmp21;
                tmp[3][m][0] = tmp30;
                tmp[3][m][1] = tmp31;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }
            for (int m = 5; m < 6; m++)
            {
                int tmp02a0 = r1[0] + r2[0];
                int tmp02a1 = r1[1] + r2[1];
                int tmp02b0 = r3[0] + r4[0];
                int tmp02b1 = r3[1] + r4[1];
                int tmp13a0 = r1[0] - r2[0];
                int tmp13a1 = r1[1] - r2[1];
                int tmp13b0 = r3[0] - r4[0];
                int tmp13b1 = r3[1] - r4[1];

                int tmp00 = tmp02a0 + tmp02b0 + r0[0];
                int tmp01 = tmp02a1 + tmp02b1 + r0[1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + r5[0] * 4;
                int tmp31 = tmp13a1 + tmp13b1 * 8 + r5[1] * 4;

                tmp00 = tmp00 * 4;
                tmp01 = tmp01 * 4;
                tmp10 = tmp10 * 4;
                tmp11 = tmp11 * 4;
                tmp20 = tmp20 * 4;
                tmp21 = tmp21 * 4;
                tmp30 = tmp30 * 4;
                tmp31 = tmp31 * 4;

                tmp[0][m][0] = tmp00;
                tmp[0][m][1] = tmp01;
                tmp[1][m][0] = tmp10;
                tmp[1][m][1] = tmp11;
                tmp[2][m][0] = tmp20;
                tmp[2][m][1] = tmp21;
                tmp[3][m][0] = tmp30;
                tmp[3][m][1] = tmp31;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                int tmp02a0 = tmp[m][1][0] + tmp[m][2][0];
                int tmp02a1 = tmp[m][1][1] + tmp[m][2][1];
                int tmp02b0 = tmp[m][3][0] + tmp[m][4][0];
                int tmp02b1 = tmp[m][3][1] + tmp[m][4][1];
                int tmp13a0 = tmp[m][1][0] - tmp[m][2][0];
                int tmp13a1 = tmp[m][1][1] - tmp[m][2][1];
                int tmp13b0 = tmp[m][3][0] - tmp[m][4][0];
                int tmp13b1 = tmp[m][3][1] - tmp[m][4][1];

                int tmp00 = tmp02a0 + tmp02b0 + tmp[m][0][0];
                int tmp01 = tmp02a1 + tmp02b1 + tmp[m][0][1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + tmp[m][5][0];
                int tmp31 = tmp13a1 + tmp13b1 * 8 + tmp[m][5][1];

                tmp00 = tmp00 / 576;
                tmp01 = tmp01 / 576;
                tmp10 = tmp10 / 576;
                tmp11 = tmp11 / 576;
                tmp20 = tmp20 / 576;
                tmp21 = tmp21 / 576;
                tmp30 = tmp30 / 576;
                tmp31 = tmp31 / 576;

                // if (out_elempack == 1)
                {
                    int* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp20;
                        outptr1[2] = tmp21;
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp30;
                        outptr1[3] = tmp31;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        int tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj;
            const int* r1 = r0 + max_jj;
            const int* r2 = r0 + max_jj * 2;
            const int* r3 = r0 + max_jj * 3;
            const int* r4 = r0 + max_jj * 4;
            const int* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 5; m++)
            {
                int tmp02a = r1[0] + r2[0];
                int tmp02b = r3[0] + r4[0];
                int tmp13a = r1[0] - r2[0];
                int tmp13b = r3[0] - r4[0];

                int tmp0 = tmp02a + tmp02b + r0[0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + r5[0] * 4;

                tmp[0][m] = tmp0;
                tmp[1][m] = tmp1;
                tmp[2][m] = tmp2;
                tmp[3][m] = tmp3;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }
            for (int m = 5; m < 6; m++)
            {
                int tmp02a = r1[0] + r2[0];
                int tmp02b = r3[0] + r4[0];
                int tmp13a = r1[0] - r2[0];
                int tmp13b = r3[0] - r4[0];

                int tmp0 = tmp02a + tmp02b + r0[0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + r5[0] * 4;

                tmp0 = tmp0 * 4;
                tmp1 = tmp1 * 4;
                tmp2 = tmp2 * 4;
                tmp3 = tmp3 * 4;

                tmp[0][m] = tmp0;
                tmp[1][m] = tmp1;
                tmp[2][m] = tmp2;
                tmp[3][m] = tmp3;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                int tmp02a = tmp[m][1] + tmp[m][2];
                int tmp02b = tmp[m][3] + tmp[m][4];
                int tmp13a = tmp[m][1] - tmp[m][2];
                int tmp13b = tmp[m][3] - tmp[m][4];

                int tmp0 = tmp02a + tmp02b + tmp[m][0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + tmp[m][5];

                tmp0 = tmp0 / 576;
                tmp1 = tmp1 / 576;
                tmp2 = tmp2 / 576;
                tmp3 = tmp3 / 576;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 4 + 1 < outw) outptr0[1] = tmp1;
                    if (tj * 4 + 2 < outw) outptr0[2] = tmp2;
                    if (tj * 4 + 3 < outw) outptr0[3] = tmp3;
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd43_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        conv3x3s1_winograd43_int8_avx512vnni(bottom_blob, top_blob, AT, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        conv3x3s1_winograd43_int8_avxvnni(bottom_blob, top_blob, AT, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        conv3x3s1_winograd43_int8_avx2(bottom_blob, top_blob, AT, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
    if (ncnn::cpu_support_x86_xop())
    {
        conv3x3s1_winograd43_int8_xop(bottom_blob, top_blob, AT, nT, opt);
        return;
    }
#endif
#endif

    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 4n+2, winograd F(4,3)
    int w_tiles = (outw + 3) / 4;
    int h_tiles = (outh + 3) / 4;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 36;

    // NCNN_LOGE("conv3x3s1_winograd43_int8 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd43_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd43_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                bool k_end = k + TILE_K >= K;

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, k_end);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile_int8(top_tile, top_blob, i, max_ii, j, max_jj);
        }
    }
}

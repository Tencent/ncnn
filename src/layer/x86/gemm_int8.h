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

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void pack_A_tile_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_A_tile_fp32_to_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void unpack_output_tile_int32_to_fp32_avx2(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose);
void gemm_transB_packed_tile_int8_avx2(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void gemm_transB_packed_tile_int8_xop(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

static void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
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
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;
        const signed char* p4 = A.row<const signed char>(i + ii + 4) + k;
        const signed char* p5 = A.row<const signed char>(i + ii + 5) + k;
        const signed char* p6 = A.row<const signed char>(i + ii + 6) + k;
        const signed char* p7 = A.row<const signed char>(i + ii + 7) + k;

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
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
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
        }
        for (; kk < max_kk; kk++)
        {
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
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
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
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;

        int kk = 0;
#if __SSE2__
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
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
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
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
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
            pp[8] = p0[4];
            pp[9] = p0[A_hstep + 4];
            pp[10] = p0[5];
            pp[11] = p0[A_hstep + 5];
            pp[12] = p0[6];
            pp[13] = p0[A_hstep + 6];
            pp[14] = p0[7];
            pp[15] = p0[A_hstep + 7];
            pp += 16;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p0[4];
            pp[5] = p0[5];
            pp[6] = p0[6];
            pp[7] = p0[7];
            pp += 8;
            p0 += A_hstep;
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
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
    // NCNN_LOGE("pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;
        const signed char* p4 = B.row<const signed char>(j + jj + 4) + k;
        const signed char* p5 = B.row<const signed char>(j + jj + 5) + k;
        const signed char* p6 = B.row<const signed char>(j + jj + 6) + k;
        const signed char* p7 = B.row<const signed char>(j + jj + 7) + k;

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
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
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
        }
        for (; kk < max_kk; kk++)
        {
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
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
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
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;

        int kk = 0;
#if __SSE2__
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
    // NCNN_LOGE("transpose_pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    const int B_hstep = B.w;

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
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
            pp[8] = p0[4];
            pp[9] = p0[B_hstep + 4];
            pp[10] = p0[5];
            pp[11] = p0[B_hstep + 5];
            pp[12] = p0[6];
            pp[13] = p0[B_hstep + 6];
            pp[14] = p0[7];
            pp[15] = p0[B_hstep + 7];
            pp += 16;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p0[4];
            pp[5] = p0[5];
            pp[6] = p0[6];
            pp[7] = p0[7];
            pp += 8;
            p0 += B_hstep;
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
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
    const int K = A.w;

    NCNN_LOGE("compute_A_tile_int8_scales %d %d", max_ii, elempack);

    const float v127_B_scale = 127.f * B_scale;

    float* ps = (float*)scales + i;
    float* pods = (float*)out_descales + i;

#if __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        __m256 _v127 = _mm256_set1_ps(127.f);
        __m256 _v127_B_scale = _mm256_set1_ps(v127_B_scale);
        for (int ii = 0; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            __m256 _absmax0 = _mm256_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m256 _p = _mm256_loadu_ps(p0);
                _absmax0 = _mm256_max_ps(_absmax0, abs256_ps(_p));
                p0 += 8;
            }

            __m256 _scale = _mm256_div_ps(_v127, _absmax0);
            __m256 _out_descale = _mm256_div_ps(_absmax0, _v127_B_scale);

            _mm256_store_ps(ps, _scale);
            _mm256_store_ps(pods, _out_descale);

            ps += 8;
            pods += 8;
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        __m128 _v127 = _mm_set1_ps(127.f);
        __m128 _v127_B_scale = _mm_set1_ps(v127_B_scale);
        for (int ii = 0; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            __m128 _absmax0 = _mm_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m128 _p = _mm_loadu_ps(p0);
                _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p));
                p0 += 4;
            }

            __m128 _scale = _mm_div_ps(_v127, _absmax0);
            __m128 _out_descale = _mm_div_ps(_absmax0, _v127_B_scale);

            _mm_store_ps(ps, _scale);
            _mm_store_ps(pods, _out_descale);

            ps += 4;
            pods += 4;
        }
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        for (int ii = 0; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            float absmax = 0.f;
            int kk = 0;
            for (; kk < K; kk++)
            {
                absmax = std::max(absmax, (float)fabsf(p0[0]));
                p0++;
            }

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
}

static void pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_A_tile_fp32_to_int8_avx2(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    NCNN_LOGE("pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    int ii = 0;
#if __SSE2__
#if __AVX__
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __AVX2__
        signed char* pp = (signed char*)AT + ii * max_kk;
#else
        signed char* pp = (signed char*)AT + ii * max_kk;
        signed char* pp1 = (signed char*)AT + (ii + 4) * max_kk;
        // NCNN_LOGE("pp0 %p", pp);
        // NCNN_LOGE("pp1 %p", pp1);
#endif

        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];

        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[8] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[10] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[11] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[4] * scale4);
                pp[9] = float2int8(p0[12] * scale4);
                pp[10] = float2int8(p0[5] * scale5);
                pp[11] = float2int8(p0[13] * scale5);
                pp[12] = float2int8(p0[6] * scale6);
                pp[13] = float2int8(p0[14] * scale6);
                pp[14] = float2int8(p0[7] * scale7);
                pp[15] = float2int8(p0[15] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[4] * scale4);
                pp1[1] = float2int8(p0[12] * scale4);
                pp1[2] = float2int8(p0[5] * scale5);
                pp1[3] = float2int8(p0[13] * scale5);
                pp1[4] = float2int8(p0[6] * scale6);
                pp1[5] = float2int8(p0[14] * scale6);
                pp1[6] = float2int8(p0[7] * scale7);
                pp1[7] = float2int8(p0[15] * scale7);
                // NCNN_LOGE("%d %d", pp[0], pp[4]);
                // NCNN_LOGE("%d %d", pp1[0], pp1[4]);
                pp += 8;
                pp1 += 8;
#endif
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
#if __AVX2__
                pp[4] = float2int8(p0[4] * scale4);
                pp[5] = float2int8(p0[5] * scale5);
                pp[6] = float2int8(p0[6] * scale6);
                pp[7] = float2int8(p0[7] * scale7);
                pp += 8;
#else
                pp1[0] = float2int8(p0[4] * scale4);
                pp1[1] = float2int8(p0[5] * scale5);
                pp1[2] = float2int8(p0[6] * scale6);
                pp1[3] = float2int8(p0[7] * scale7);
                pp += 4;
                pp1 += 4;
#endif
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[4] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[6] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[7] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[A_hstep * 4 + 0] * scale4);
                pp[9] = float2int8(p0[A_hstep * 4 + 4] * scale4);
                pp[10] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp[11] = float2int8(p0[A_hstep * 4 + 5] * scale5);
                pp[12] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp[13] = float2int8(p0[A_hstep * 4 + 6] * scale6);
                pp[14] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp[15] = float2int8(p0[A_hstep * 4 + 7] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[A_hstep * 4 + 0] * scale4);
                pp1[1] = float2int8(p0[A_hstep * 4 + 4] * scale4);
                pp1[2] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp1[3] = float2int8(p0[A_hstep * 4 + 5] * scale5);
                pp1[4] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp1[5] = float2int8(p0[A_hstep * 4 + 6] * scale6);
                pp1[6] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp1[7] = float2int8(p0[A_hstep * 4 + 7] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
#if __AVX2__
                pp[4] = float2int8(p0[A_hstep * 4] * scale4);
                pp[5] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp[6] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp[7] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp += 8;
#else
                pp1[0] = float2int8(p0[A_hstep * 4] * scale4);
                pp1[1] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp1[2] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp1[3] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp += 4;
                pp1 += 4;
#endif
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[A_hstep] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[A_hstep * 2] * scale2);
                pp[5] = float2int8(p0[A_hstep * 2 + 1] * scale2);
                pp[6] = float2int8(p0[A_hstep * 3] * scale3);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[A_hstep * 4] * scale4);
                pp[9] = float2int8(p0[A_hstep * 4 + 1] * scale4);
                pp[10] = float2int8(p0[A_hstep * 5] * scale5);
                pp[11] = float2int8(p0[A_hstep * 5 + 1] * scale5);
                pp[12] = float2int8(p0[A_hstep * 6] * scale6);
                pp[13] = float2int8(p0[A_hstep * 6 + 1] * scale6);
                pp[14] = float2int8(p0[A_hstep * 7] * scale7);
                pp[15] = float2int8(p0[A_hstep * 7 + 1] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[A_hstep * 4] * scale4);
                pp1[1] = float2int8(p0[A_hstep * 4 + 1] * scale4);
                pp1[2] = float2int8(p0[A_hstep * 5] * scale5);
                pp1[3] = float2int8(p0[A_hstep * 5 + 1] * scale5);
                pp1[4] = float2int8(p0[A_hstep * 6] * scale6);
                pp1[5] = float2int8(p0[A_hstep * 6 + 1] * scale6);
                pp1[6] = float2int8(p0[A_hstep * 7] * scale7);
                pp1[7] = float2int8(p0[A_hstep * 7 + 1] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale1);
                pp[2] = float2int8(p0[A_hstep * 2] * scale2);
                pp[3] = float2int8(p0[A_hstep * 3] * scale3);
#if __AVX2__
                pp[4] = float2int8(p0[A_hstep * 4] * scale4);
                pp[5] = float2int8(p0[A_hstep * 5] * scale5);
                pp[6] = float2int8(p0[A_hstep * 6] * scale6);
                pp[7] = float2int8(p0[A_hstep * 7] * scale7);
                pp += 8;
#else
                pp1[0] = float2int8(p0[A_hstep * 4] * scale4);
                pp1[1] = float2int8(p0[A_hstep * 5] * scale5);
                pp1[2] = float2int8(p0[A_hstep * 6] * scale6);
                pp1[3] = float2int8(p0[A_hstep * 7] * scale7);
                pp += 4;
                pp1 += 4;
#endif
                p0++;
            }
        }
    }
    signed char* pp = (signed char*)AT + ii * max_kk;
#else
    signed char* pp = (signed char*)AT;
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[4] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[6] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[7] * scale3);

                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);

                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[A_hstep] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[A_hstep * 2] * scale2);
                pp[5] = float2int8(p0[A_hstep * 2 + 1] * scale2);
                pp[6] = float2int8(p0[A_hstep * 3] * scale3);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale3);

                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale1);
                pp[2] = float2int8(p0[A_hstep * 2] * scale2);
                pp[3] = float2int8(p0[A_hstep * 3] * scale3);

                pp += 4;
                p0++;
            }
        }
    }
#else
    signed char* pp = (signed char*)AT;
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        // if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[A_hstep] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
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

        // if (elempack == 1)
        {
            int kk = 0;
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
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
    const int K = A.dims == 3 ? A.c : A.h;

    NCNN_LOGE("transpose_compute_A_tile_int8_scales %d %d", max_ii, elempack);

    const float v127_B_scale = 127.f * B_scale;

    float* ps = (float*)scales + i;
    float* pods = (float*)out_descales + i;

#if __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        int ii = 0;
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * 8;

            __m256 _absmax0 = _mm256_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m256 _p = _mm256_loadu_ps(p0);
                _absmax0 = _mm256_max_ps(_absmax0, abs256_ps(_p));
                p0 += A_hstep * 8;
            }
            float absmax = _mm256_reduce_max_ps(_absmax0);

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        int ii = 0;
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * 4;

            __m128 _absmax0 = _mm_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m128 _p = _mm_loadu_ps(p0);
                _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p));
                p0 += A_hstep * 4;
            }
            float absmax = _mm_reduce_max_ps(_absmax0);

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        int ii = 0;
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii);

            float absmax = 0.f;
            for (int kk = 0; kk < K; kk++)
            {
                absmax = std::max(absmax, (float)fabsf(p0[0]));
                p0 += A_hstep;
            }

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_pack_A_tile_fp32_to_int8_avx2(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    NCNN_LOGE("transpose_pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    int ii = 0;
#if __SSE2__
#if __AVX__
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __AVX2__
        signed char* pp = (signed char*)AT + ii * max_kk;
#else
        signed char* pp = (signed char*)AT + ii * max_kk;
        signed char* pp1 = (signed char*)AT + (ii + 4) * max_kk;
#endif

        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];

        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[8] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[16] * scale2);
                pp[5] = float2int8(p0[17] * scale2);
                pp[6] = float2int8(p0[24] * scale3);
                pp[7] = float2int8(p0[25] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[32] * scale4);
                pp[9] = float2int8(p0[33] * scale4);
                pp[10] = float2int8(p0[40] * scale5);
                pp[11] = float2int8(p0[41] * scale5);
                pp[12] = float2int8(p0[48] * scale6);
                pp[13] = float2int8(p0[49] * scale6);
                pp[14] = float2int8(p0[56] * scale7);
                pp[15] = float2int8(p0[57] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[32] * scale4);
                pp1[1] = float2int8(p0[33] * scale4);
                pp1[2] = float2int8(p0[40] * scale5);
                pp1[3] = float2int8(p0[41] * scale5);
                pp1[4] = float2int8(p0[48] * scale6);
                pp1[5] = float2int8(p0[49] * scale6);
                pp1[6] = float2int8(p0[56] * scale7);
                pp1[7] = float2int8(p0[57] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                pp[0] = float2int8(p0[2] * scale0);
                pp[1] = float2int8(p0[3] * scale0);
                pp[2] = float2int8(p0[10] * scale1);
                pp[3] = float2int8(p0[11] * scale1);
                pp[4] = float2int8(p0[18] * scale2);
                pp[5] = float2int8(p0[19] * scale2);
                pp[6] = float2int8(p0[26] * scale3);
                pp[7] = float2int8(p0[27] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[34] * scale4);
                pp[9] = float2int8(p0[35] * scale4);
                pp[10] = float2int8(p0[42] * scale5);
                pp[11] = float2int8(p0[43] * scale5);
                pp[12] = float2int8(p0[50] * scale6);
                pp[13] = float2int8(p0[51] * scale6);
                pp[14] = float2int8(p0[58] * scale7);
                pp[15] = float2int8(p0[59] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[34] * scale4);
                pp1[1] = float2int8(p0[35] * scale4);
                pp1[2] = float2int8(p0[42] * scale5);
                pp1[3] = float2int8(p0[43] * scale5);
                pp1[4] = float2int8(p0[50] * scale6);
                pp1[5] = float2int8(p0[51] * scale6);
                pp1[6] = float2int8(p0[58] * scale7);
                pp1[7] = float2int8(p0[59] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                pp[0] = float2int8(p0[4] * scale0);
                pp[1] = float2int8(p0[5] * scale0);
                pp[2] = float2int8(p0[12] * scale1);
                pp[3] = float2int8(p0[13] * scale1);
                pp[4] = float2int8(p0[20] * scale2);
                pp[5] = float2int8(p0[21] * scale2);
                pp[6] = float2int8(p0[28] * scale3);
                pp[7] = float2int8(p0[29] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[36] * scale4);
                pp[9] = float2int8(p0[37] * scale4);
                pp[10] = float2int8(p0[44] * scale5);
                pp[11] = float2int8(p0[45] * scale5);
                pp[12] = float2int8(p0[52] * scale6);
                pp[13] = float2int8(p0[53] * scale6);
                pp[14] = float2int8(p0[60] * scale7);
                pp[15] = float2int8(p0[61] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[36] * scale4);
                pp1[1] = float2int8(p0[37] * scale4);
                pp1[2] = float2int8(p0[44] * scale5);
                pp1[3] = float2int8(p0[45] * scale5);
                pp1[4] = float2int8(p0[52] * scale6);
                pp1[5] = float2int8(p0[53] * scale6);
                pp1[6] = float2int8(p0[60] * scale7);
                pp1[7] = float2int8(p0[61] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                pp[0] = float2int8(p0[6] * scale0);
                pp[1] = float2int8(p0[7] * scale0);
                pp[2] = float2int8(p0[14] * scale1);
                pp[3] = float2int8(p0[15] * scale1);
                pp[4] = float2int8(p0[22] * scale2);
                pp[5] = float2int8(p0[23] * scale2);
                pp[6] = float2int8(p0[30] * scale3);
                pp[7] = float2int8(p0[31] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[38] * scale4);
                pp[9] = float2int8(p0[39] * scale4);
                pp[10] = float2int8(p0[46] * scale5);
                pp[11] = float2int8(p0[47] * scale5);
                pp[12] = float2int8(p0[54] * scale6);
                pp[13] = float2int8(p0[55] * scale6);
                pp[14] = float2int8(p0[62] * scale7);
                pp[15] = float2int8(p0[63] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[38] * scale4);
                pp1[1] = float2int8(p0[39] * scale4);
                pp1[2] = float2int8(p0[46] * scale5);
                pp1[3] = float2int8(p0[47] * scale5);
                pp1[4] = float2int8(p0[54] * scale6);
                pp1[5] = float2int8(p0[55] * scale6);
                pp1[6] = float2int8(p0[62] * scale7);
                pp1[7] = float2int8(p0[63] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[4] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[8] * scale2);
                pp[5] = float2int8(p0[9] * scale2);
                pp[6] = float2int8(p0[12] * scale3);
                pp[7] = float2int8(p0[13] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[16] * scale4);
                pp[9] = float2int8(p0[17] * scale4);
                pp[10] = float2int8(p0[20] * scale5);
                pp[11] = float2int8(p0[21] * scale5);
                pp[12] = float2int8(p0[24] * scale6);
                pp[13] = float2int8(p0[25] * scale6);
                pp[14] = float2int8(p0[28] * scale7);
                pp[15] = float2int8(p0[29] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[16] * scale4);
                pp1[1] = float2int8(p0[17] * scale4);
                pp1[2] = float2int8(p0[20] * scale5);
                pp1[3] = float2int8(p0[21] * scale5);
                pp1[4] = float2int8(p0[24] * scale6);
                pp1[5] = float2int8(p0[25] * scale6);
                pp1[6] = float2int8(p0[28] * scale7);
                pp1[7] = float2int8(p0[29] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                pp[0] = float2int8(p0[2] * scale0);
                pp[1] = float2int8(p0[3] * scale0);
                pp[2] = float2int8(p0[6] * scale1);
                pp[3] = float2int8(p0[7] * scale1);
                pp[4] = float2int8(p0[10] * scale2);
                pp[5] = float2int8(p0[11] * scale2);
                pp[6] = float2int8(p0[14] * scale3);
                pp[7] = float2int8(p0[15] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[18] * scale4);
                pp[9] = float2int8(p0[19] * scale4);
                pp[10] = float2int8(p0[22] * scale5);
                pp[11] = float2int8(p0[23] * scale5);
                pp[12] = float2int8(p0[26] * scale6);
                pp[13] = float2int8(p0[27] * scale6);
                pp[14] = float2int8(p0[30] * scale7);
                pp[15] = float2int8(p0[31] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[18] * scale4);
                pp1[1] = float2int8(p0[19] * scale4);
                pp1[2] = float2int8(p0[22] * scale5);
                pp1[3] = float2int8(p0[23] * scale5);
                pp1[4] = float2int8(p0[26] * scale6);
                pp1[5] = float2int8(p0[27] * scale6);
                pp1[6] = float2int8(p0[30] * scale7);
                pp1[7] = float2int8(p0[31] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[A_hstep + 2] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[A_hstep + 3] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[4] * scale4);
                pp[9] = float2int8(p0[A_hstep + 4] * scale4);
                pp[10] = float2int8(p0[5] * scale5);
                pp[11] = float2int8(p0[A_hstep + 5] * scale5);
                pp[12] = float2int8(p0[6] * scale6);
                pp[13] = float2int8(p0[A_hstep + 6] * scale6);
                pp[14] = float2int8(p0[7] * scale7);
                pp[15] = float2int8(p0[A_hstep + 7] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[4] * scale4);
                pp1[1] = float2int8(p0[A_hstep + 4] * scale4);
                pp1[2] = float2int8(p0[5] * scale5);
                pp1[3] = float2int8(p0[A_hstep + 5] * scale5);
                pp1[4] = float2int8(p0[6] * scale6);
                pp1[5] = float2int8(p0[A_hstep + 6] * scale6);
                pp1[6] = float2int8(p0[7] * scale7);
                pp1[7] = float2int8(p0[A_hstep + 7] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
#if __AVX2__
                pp[4] = float2int8(p0[4] * scale4);
                pp[5] = float2int8(p0[5] * scale5);
                pp[6] = float2int8(p0[6] * scale6);
                pp[7] = float2int8(p0[7] * scale7);
                pp += 8;
#else
                pp1[0] = float2int8(p0[4] * scale4);
                pp1[1] = float2int8(p0[5] * scale5);
                pp1[2] = float2int8(p0[6] * scale6);
                pp1[3] = float2int8(p0[7] * scale7);
                pp += 4;
                pp1 += 4;
#endif
                p0 += A_hstep;
            }
        }
    }
    signed char* pp = (signed char*)AT + ii * max_kk;
#else
    signed char* pp = (signed char*)AT;
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];

#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[8] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[16] * scale2);
                pp[5] = float2int8(p0[17] * scale2);
                pp[6] = float2int8(p0[24] * scale3);
                pp[7] = float2int8(p0[25] * scale3);

                pp[8] = float2int8(p0[2] * scale0);
                pp[9] = float2int8(p0[3] * scale0);
                pp[10] = float2int8(p0[10] * scale1);
                pp[11] = float2int8(p0[11] * scale1);
                pp[12] = float2int8(p0[18] * scale2);
                pp[13] = float2int8(p0[19] * scale2);
                pp[14] = float2int8(p0[26] * scale3);
                pp[15] = float2int8(p0[27] * scale3);

                pp[16 + 0] = float2int8(p0[4] * scale0);
                pp[16 + 1] = float2int8(p0[5] * scale0);
                pp[16 + 2] = float2int8(p0[12] * scale1);
                pp[16 + 3] = float2int8(p0[13] * scale1);
                pp[16 + 4] = float2int8(p0[20] * scale2);
                pp[16 + 5] = float2int8(p0[21] * scale2);
                pp[16 + 6] = float2int8(p0[28] * scale3);
                pp[16 + 7] = float2int8(p0[29] * scale3);

                pp[16 + 8] = float2int8(p0[6] * scale0);
                pp[16 + 9] = float2int8(p0[7] * scale0);
                pp[16 + 10] = float2int8(p0[14] * scale1);
                pp[16 + 11] = float2int8(p0[15] * scale1);
                pp[16 + 12] = float2int8(p0[22] * scale2);
                pp[16 + 13] = float2int8(p0[23] * scale2);
                pp[16 + 14] = float2int8(p0[30] * scale3);
                pp[16 + 15] = float2int8(p0[31] * scale3);

                pp += 32;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[4] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[8] * scale2);
                pp[5] = float2int8(p0[9] * scale2);
                pp[6] = float2int8(p0[12] * scale3);
                pp[7] = float2int8(p0[13] * scale3);
                pp[8] = float2int8(p0[2] * scale0);
                pp[9] = float2int8(p0[3] * scale0);
                pp[10] = float2int8(p0[6] * scale1);
                pp[11] = float2int8(p0[7] * scale1);
                pp[12] = float2int8(p0[10] * scale2);
                pp[13] = float2int8(p0[11] * scale2);
                pp[14] = float2int8(p0[14] * scale3);
                pp[15] = float2int8(p0[15] * scale3);

                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[A_hstep + 2] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[A_hstep + 3] * scale3);

                pp += 8;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);

                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#else
    signed char* pp = (signed char*)AT;
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

#if __SSE2__
#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[8] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[2] * scale0);
                pp[5] = float2int8(p0[3] * scale0);
                pp[6] = float2int8(p0[10] * scale1);
                pp[7] = float2int8(p0[11] * scale1);
                pp[8] = float2int8(p0[4] * scale0);
                pp[9] = float2int8(p0[5] * scale0);
                pp[10] = float2int8(p0[12] * scale1);
                pp[11] = float2int8(p0[13] * scale1);
                pp[12] = float2int8(p0[6] * scale0);
                pp[13] = float2int8(p0[7] * scale0);
                pp[14] = float2int8(p0[14] * scale1);
                pp[15] = float2int8(p0[15] * scale1);

                pp += 16;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[4] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[2] * scale0);
                pp[5] = float2int8(p0[3] * scale0);
                pp[6] = float2int8(p0[6] * scale1);
                pp[7] = float2int8(p0[7] * scale1);

                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
#if __SSE2__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep + 0] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
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

        const float scale = scales[i + ii];

#if __SSE2__
#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
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
    NCNN_LOGE("compute_B_fp32_int8_scale");

    float absmax = 0.f;
#if __SSE2__
#if __AVX__
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
    absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax_avx));
#endif // __AVX__
    absmax = std::max(absmax, _mm_reduce_max_ps(_absmax));
#endif

    scale = absmax == 0.f ? 1.f : 127.f / absmax;
}

static void pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    NCNN_LOGE("pack_B_tile_fp32_to_int8 %d %d %d", max_jj, max_kk, elempack);

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[8] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[10] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[11] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[12] * scale);
                pp[10] = float2int8(p0[5] * scale);
                pp[11] = float2int8(p0[13] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[14] * scale);
                pp[14] = float2int8(p0[7] * scale);
                pp[15] = float2int8(p0[15] * scale);

                pp += 16;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);

                pp += 8;
                p0 += 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[4] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[6] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp[8] = float2int8(p0[B_hstep * 4] * scale);
                pp[9] = float2int8(p0[B_hstep * 4 + 4] * scale);
                pp[10] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[11] = float2int8(p0[B_hstep * 4 + 5] * scale);
                pp[12] = float2int8(p0[B_hstep * 4 + 2] * scale);
                pp[13] = float2int8(p0[B_hstep * 4 + 6] * scale);
                pp[14] = float2int8(p0[B_hstep * 4 + 3] * scale);
                pp[15] = float2int8(p0[B_hstep * 4 + 7] * scale);

                pp += 16;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[B_hstep * 4] * scale);
                pp[5] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[6] = float2int8(p0[B_hstep * 4 + 2] * scale);
                pp[7] = float2int8(p0[B_hstep * 4 + 3] * scale);

                pp += 8;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[B_hstep] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[B_hstep * 2] * scale);
                pp[5] = float2int8(p0[B_hstep * 2 + 1] * scale);
                pp[6] = float2int8(p0[B_hstep * 3] * scale);
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale);
                pp[8] = float2int8(p0[B_hstep * 4] * scale);
                pp[9] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[10] = float2int8(p0[B_hstep * 5] * scale);
                pp[11] = float2int8(p0[B_hstep * 5 + 1] * scale);
                pp[12] = float2int8(p0[B_hstep * 6] * scale);
                pp[13] = float2int8(p0[B_hstep * 6 + 1] * scale);
                pp[14] = float2int8(p0[B_hstep * 7] * scale);
                pp[15] = float2int8(p0[B_hstep * 7 + 1] * scale);

                pp += 16;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[B_hstep * 2] * scale);
                pp[3] = float2int8(p0[B_hstep * 3] * scale);
                pp[4] = float2int8(p0[B_hstep * 4] * scale);
                pp[5] = float2int8(p0[B_hstep * 5] * scale);
                pp[6] = float2int8(p0[B_hstep * 6] * scale);
                pp[7] = float2int8(p0[B_hstep * 7] * scale);

                pp += 8;
                p0++;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[4] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[6] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[7] * scale);

                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);

                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[B_hstep] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[B_hstep * 2] * scale);
                pp[5] = float2int8(p0[B_hstep * 2 + 1] * scale);
                pp[6] = float2int8(p0[B_hstep * 3] * scale);
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale);

                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[B_hstep * 2] * scale);
                pp[3] = float2int8(p0[B_hstep * 3] * scale);

                pp += 4;
                p0++;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        // if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[B_hstep] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
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

        // if (elempack == 1)
        {
            int kk = 0;
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
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    NCNN_LOGE("transpose_pack_B_tile_fp32_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[8] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[16] * scale);
                pp[5] = float2int8(p0[17] * scale);
                pp[6] = float2int8(p0[24] * scale);
                pp[7] = float2int8(p0[25] * scale);
                pp[8] = float2int8(p0[32] * scale);
                pp[9] = float2int8(p0[33] * scale);
                pp[10] = float2int8(p0[40] * scale);
                pp[11] = float2int8(p0[41] * scale);
                pp[12] = float2int8(p0[48] * scale);
                pp[13] = float2int8(p0[49] * scale);
                pp[14] = float2int8(p0[56] * scale);
                pp[15] = float2int8(p0[57] * scale);
                pp += 16;

                pp[0] = float2int8(p0[2] * scale);
                pp[1] = float2int8(p0[3] * scale);
                pp[2] = float2int8(p0[10] * scale);
                pp[3] = float2int8(p0[11] * scale);
                pp[4] = float2int8(p0[18] * scale);
                pp[5] = float2int8(p0[19] * scale);
                pp[6] = float2int8(p0[26] * scale);
                pp[7] = float2int8(p0[27] * scale);
                pp[8] = float2int8(p0[34] * scale);
                pp[9] = float2int8(p0[35] * scale);
                pp[10] = float2int8(p0[42] * scale);
                pp[11] = float2int8(p0[43] * scale);
                pp[12] = float2int8(p0[50] * scale);
                pp[13] = float2int8(p0[51] * scale);
                pp[14] = float2int8(p0[58] * scale);
                pp[15] = float2int8(p0[59] * scale);
                pp += 16;

                pp[0] = float2int8(p0[4] * scale);
                pp[1] = float2int8(p0[5] * scale);
                pp[2] = float2int8(p0[12] * scale);
                pp[3] = float2int8(p0[13] * scale);
                pp[4] = float2int8(p0[20] * scale);
                pp[5] = float2int8(p0[21] * scale);
                pp[6] = float2int8(p0[28] * scale);
                pp[7] = float2int8(p0[29] * scale);
                pp[8] = float2int8(p0[36] * scale);
                pp[9] = float2int8(p0[37] * scale);
                pp[10] = float2int8(p0[44] * scale);
                pp[11] = float2int8(p0[45] * scale);
                pp[12] = float2int8(p0[52] * scale);
                pp[13] = float2int8(p0[53] * scale);
                pp[14] = float2int8(p0[60] * scale);
                pp[15] = float2int8(p0[61] * scale);
                pp += 16;

                pp[0] = float2int8(p0[6] * scale);
                pp[1] = float2int8(p0[7] * scale);
                pp[2] = float2int8(p0[14] * scale);
                pp[3] = float2int8(p0[15] * scale);
                pp[4] = float2int8(p0[22] * scale);
                pp[5] = float2int8(p0[23] * scale);
                pp[6] = float2int8(p0[30] * scale);
                pp[7] = float2int8(p0[31] * scale);
                pp[8] = float2int8(p0[38] * scale);
                pp[9] = float2int8(p0[39] * scale);
                pp[10] = float2int8(p0[46] * scale);
                pp[11] = float2int8(p0[47] * scale);
                pp[12] = float2int8(p0[54] * scale);
                pp[13] = float2int8(p0[55] * scale);
                pp[14] = float2int8(p0[62] * scale);
                pp[15] = float2int8(p0[63] * scale);
                pp += 16;

                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);
                pp[8] = float2int8(p0[16] * scale);
                pp[9] = float2int8(p0[17] * scale);
                pp[10] = float2int8(p0[20] * scale);
                pp[11] = float2int8(p0[21] * scale);
                pp[12] = float2int8(p0[24] * scale);
                pp[13] = float2int8(p0[25] * scale);
                pp[14] = float2int8(p0[28] * scale);
                pp[15] = float2int8(p0[29] * scale);

                pp[16 + 0] = float2int8(p0[2] * scale);
                pp[16 + 1] = float2int8(p0[3] * scale);
                pp[16 + 2] = float2int8(p0[6] * scale);
                pp[16 + 3] = float2int8(p0[7] * scale);
                pp[16 + 4] = float2int8(p0[10] * scale);
                pp[16 + 5] = float2int8(p0[11] * scale);
                pp[16 + 6] = float2int8(p0[14] * scale);
                pp[16 + 7] = float2int8(p0[15] * scale);
                pp[16 + 8] = float2int8(p0[18] * scale);
                pp[16 + 9] = float2int8(p0[19] * scale);
                pp[16 + 10] = float2int8(p0[22] * scale);
                pp[16 + 11] = float2int8(p0[23] * scale);
                pp[16 + 12] = float2int8(p0[26] * scale);
                pp[16 + 13] = float2int8(p0[27] * scale);
                pp[16 + 14] = float2int8(p0[30] * scale);
                pp[16 + 15] = float2int8(p0[31] * scale);

                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[B_hstep + 2] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[B_hstep + 3] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[B_hstep + 4] * scale);
                pp[10] = float2int8(p0[5] * scale);
                pp[11] = float2int8(p0[B_hstep + 5] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[B_hstep + 6] * scale);
                pp[14] = float2int8(p0[7] * scale);
                pp[15] = float2int8(p0[B_hstep + 7] * scale);

                pp += 16;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[8] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[16] * scale);
                pp[5] = float2int8(p0[17] * scale);
                pp[6] = float2int8(p0[24] * scale);
                pp[7] = float2int8(p0[25] * scale);

                pp[8] = float2int8(p0[2] * scale);
                pp[9] = float2int8(p0[3] * scale);
                pp[10] = float2int8(p0[10] * scale);
                pp[11] = float2int8(p0[11] * scale);
                pp[12] = float2int8(p0[18] * scale);
                pp[13] = float2int8(p0[19] * scale);
                pp[14] = float2int8(p0[26] * scale);
                pp[15] = float2int8(p0[27] * scale);

                pp[16 + 0] = float2int8(p0[4] * scale);
                pp[16 + 1] = float2int8(p0[5] * scale);
                pp[16 + 2] = float2int8(p0[12] * scale);
                pp[16 + 3] = float2int8(p0[13] * scale);
                pp[16 + 4] = float2int8(p0[20] * scale);
                pp[16 + 5] = float2int8(p0[21] * scale);
                pp[16 + 6] = float2int8(p0[28] * scale);
                pp[16 + 7] = float2int8(p0[29] * scale);

                pp[16 + 8] = float2int8(p0[6] * scale);
                pp[16 + 9] = float2int8(p0[7] * scale);
                pp[16 + 10] = float2int8(p0[14] * scale);
                pp[16 + 11] = float2int8(p0[15] * scale);
                pp[16 + 12] = float2int8(p0[22] * scale);
                pp[16 + 13] = float2int8(p0[23] * scale);
                pp[16 + 14] = float2int8(p0[30] * scale);
                pp[16 + 15] = float2int8(p0[31] * scale);

                pp += 32;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);
                pp[8] = float2int8(p0[2] * scale);
                pp[9] = float2int8(p0[3] * scale);
                pp[10] = float2int8(p0[6] * scale);
                pp[11] = float2int8(p0[7] * scale);
                pp[12] = float2int8(p0[10] * scale);
                pp[13] = float2int8(p0[11] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);

                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[B_hstep + 2] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[B_hstep + 3] * scale);

                pp += 8;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __SSE2__
#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[8] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[3] * scale);
                pp[6] = float2int8(p0[10] * scale);
                pp[7] = float2int8(p0[11] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[5] * scale);
                pp[10] = float2int8(p0[12] * scale);
                pp[11] = float2int8(p0[13] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[7] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);

                pp += 16;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[3] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);

                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep + 0] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
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

#if __SSE2__
#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
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
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
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

    NCNN_LOGE("unpack_output_tile_int32_to_fp32  %d %d %d %d  %d  %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C, c_elempack, output_transpose);

    // const int* pp = topT;

    int ii = 0;
#if __SSE2__
#if __AVX__
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __AVX2__
        const int* pp = (const int*)topT + ii * max_jj;
#else
        const int* pp = (const int*)topT + ii * max_jj;
        const int* pp1 = (const int*)topT + (ii + 4) * max_jj;
#endif

        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m256 _descale = _mm256_loadu_ps((const float*)descales + i + ii);

        __m256 _c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm256_set1_ps(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm256_loadu_ps(pC);
                _c0 = _mm256_mul_ps(_c0, _mm256_set1_ps(beta));
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
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            __m256 _f2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 16)));
            __m256 _f3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 24)));
            __m256 _f4 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 32)));
            __m256 _f5 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 40)));
            __m256 _f6 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 48)));
            __m256 _f7 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 56)));
            pp += 64;

            // from
            //      00 11 22 33 44 55 66 77
            //      01 12 23 30 45 56 67 74
            //      60 71 42 53 24 35 06 17
            //      61 72 43 50 25 36 07 14
            //      02 13 20 31 46 57 64 75
            //      03 10 21 32 47 54 65 76
            //      62 73 40 51 26 37 04 15
            //      63 70 41 52 27 34 05 16

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
                __m256 _tmp4 = _mm256_shuffle_ps(_f4, _f4, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _tmp5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(0, 3, 2, 1));
                __m256 _tmp6 = _mm256_shuffle_ps(_f6, _f6, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _tmp7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(0, 3, 2, 1));

                // 00 11 22 33 44 55 66 77
                // 30 01 12 23 74 45 56 67
                // 60 71 42 53 24 35 06 17
                // 50 61 72 43 14 25 36 07
                // 20 31 02 13 64 75 46 57
                // 10 21 32 03 54 65 76 47
                // 40 51 62 73 04 15 26 37
                // 70 41 52 63 34 05 16 27

                _f0 = _mm256_permute2f128_ps(_tmp0, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                _f1 = _mm256_permute2f128_ps(_tmp5, _tmp3, _MM_SHUFFLE(0, 2, 0, 0));
                _f2 = _mm256_permute2f128_ps(_tmp4, _tmp2, _MM_SHUFFLE(0, 2, 0, 0));
                _f3 = _mm256_permute2f128_ps(_tmp1, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                _f4 = _mm256_permute2f128_ps(_tmp6, _tmp0, _MM_SHUFFLE(0, 3, 0, 1));
                _f5 = _mm256_permute2f128_ps(_tmp3, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                _f6 = _mm256_permute2f128_ps(_tmp2, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                _f7 = _mm256_permute2f128_ps(_tmp7, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));

                // 00 11 22 33 40 51 62 73
                // 10 21 32 03 50 61 72 43
                // 20 31 02 13 60 71 42 53
                // 30 01 12 23 70 41 52 63
                // 04 15 26 37 44 55 66 77
                // 14 25 36 07 54 65 76 47
                // 24 35 06 17 64 75 46 57
                // 34 05 16 27 74 45 56 67

                _tmp0 = _mm256_unpacklo_ps(_f0, _f1);
                _tmp1 = _mm256_unpacklo_ps(_f2, _f3);
                _tmp2 = _mm256_unpackhi_ps(_f2, _f3);
                _tmp3 = _mm256_unpackhi_ps(_f0, _f1);
                _tmp4 = _mm256_unpacklo_ps(_f4, _f5);
                _tmp5 = _mm256_unpacklo_ps(_f6, _f7);
                _tmp6 = _mm256_unpackhi_ps(_f6, _f7);
                _tmp7 = _mm256_unpackhi_ps(_f4, _f5);

                // 00 10 11 21 40 50 51 61
                // 20 30 31 01 60 70 71 41
                // 02 12 13 23 42 52 53 63
                // 22 32 33 03 62 72 73 43

                // 04 14 15 25 44 54 55 65
                // 24 34 35 05 64 74 75 45
                // 06 16 17 27 46 56 57 67
                // 26 36 37 07 66 76 77 47

                _f0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));
                _f7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));

                // 00 10 20 30 40 50 60 70
                // 11 21 31 01 51 61 71 41
                // 02 12 22 32 42 52 62 72
                // 13 23 33 03 53 63 73 43
                // 04 14 24 34 44 54 64 74
                // 15 25 35 05 55 65 75 45
                // 06 16 26 36 46 56 66 76
                // 17 27 37 07 57 67 77 47

                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
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

                // 00 11 22 33 04 15 26 37
                // 20 31 02 13 24 35 06 17
                // 30 01 12 23 34 05 16 27
                // 10 21 32 03 14 25 36 07
                // 40 51 62 73 44 55 66 77
                // 60 71 42 53 64 75 46 57
                // 70 41 52 63 74 45 56 67
                // 50 61 72 43 54 65 76 47

                _f0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 2, 0, 0));
                _f1 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                _f2 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                _f3 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                _f4 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                _f5 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                _f6 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 1));
                _f7 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                // 00 11 22 33 40 51 62 73
                // 20 31 02 13 60 71 42 53
                // 30 01 12 23 70 41 52 63
                // 10 21 32 03 50 61 72 43
                // 04 15 26 37 44 55 66 77
                // 24 35 06 17 64 75 46 57
                // 34 05 16 27 74 45 56 67
                // 14 25 36 07 54 65 76 47

                _tmp0 = _mm256_unpacklo_ps(_f0, _f3);
                _tmp1 = _mm256_unpacklo_ps(_f1, _f2);
                _tmp2 = _mm256_unpackhi_ps(_f1, _f2);
                _tmp3 = _mm256_unpackhi_ps(_f0, _f3);
                _tmp4 = _mm256_unpacklo_ps(_f4, _f7);
                _tmp5 = _mm256_unpacklo_ps(_f5, _f6);
                _tmp6 = _mm256_unpackhi_ps(_f5, _f6);
                _tmp7 = _mm256_unpackhi_ps(_f4, _f7);

                // 00 10 11 21 40 50 51 61
                // 20 30 31 01 60 70 71 41
                // 02 12 13 23 42 52 53 63
                // 22 32 33 03 62 72 73 43
                // 04 14 15 25 44 54 55 65
                // 24 34 35 05 64 74 75 45
                // 06 16 17 27 46 56 57 67
                // 26 36 37 07 66 76 77 47

                _f0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));
                _f7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));

                // 00 10 20 30 40 50 60 70
                // 11 21 31 01 51 61 71 41
                // 02 12 22 32 42 52 62 72
                // 13 23 33 03 53 63 73 43
                // 04 14 24 34 44 54 64 74
                // 15 25 35 05

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
                    if (c_elempack == 4)
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
                    if (c_elempack == 1)
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
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                    _mm256_storeu_ps(p0 + 32, _f4);
                    _mm256_storeu_ps(p0 + 40, _f5);
                    _mm256_storeu_ps(p0 + 48, _f6);
                    _mm256_storeu_ps(p0 + 56, _f7);
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
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                    _mm256_storeu_ps(p0 + 32, _f4);
                    _mm256_storeu_ps(p0 + 40, _f5);
                    _mm256_storeu_ps(p0 + 48, _f6);
                    _mm256_storeu_ps(p0 + 56, _f7);
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
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            __m256 _f2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 16)));
            __m256 _f3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 24)));
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
            //      00 11 22 33
            //      01 12 23 30
            //      20 31 02 13
            //      21 32 03 10

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
                    if (c_elempack == 4)
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
                    if (c_elempack == 1)
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

                        _c0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc4, 1);
                        _c1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc1), _cc5, 1);
                        _c2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc2), _cc6, 1);
                        _c3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc3), _cc7, 1);

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
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
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
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
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
                    if (c_elempack == 4)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep * 4);
                        _c0 = _mm256_permute2f128_ps(_cc0, _cc1, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_cc0, _cc1, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 8;
                    }
                    if (c_elempack == 1)
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

                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            pp += 8;
#else
            __m128i _f0l = _mm_loadu_si128((const __m128i*)pp);
            __m128i _f0h = _mm_loadu_si128((const __m128i*)pp1);
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_f0l), _f0h, 1));
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
                    if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4);
                        _c0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc1, 1);
                        pC += 4;
                    }
                    if (c_elempack == 1)
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
                    _mm_storeu_ps(p0, _mm256_extractf128_ps(_f0, 0));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm256_extractf128_ps(_f0, 1));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
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
                    p0++;
                }
            }
        }
    }
    const int* pp = (const int*)topT + ii * max_jj;
#else
    const int* pp = (const int*)topT;
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

        __m128 _descale = _mm_loadu_ps((const float*)descales + i + ii);

        __m128 _c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm_set1_ps(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm_loadu_ps(pC);
                _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
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
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));
            __m128i _sum2 = _mm_loadu_si128((const __m128i*)(pp + 8));
            __m128i _sum3 = _mm_loadu_si128((const __m128i*)(pp + 12));
            __m128i _sum4 = _mm_loadu_si128((const __m128i*)(pp + 16));
            __m128i _sum5 = _mm_loadu_si128((const __m128i*)(pp + 20));
            __m128i _sum6 = _mm_loadu_si128((const __m128i*)(pp + 24));
            __m128i _sum7 = _mm_loadu_si128((const __m128i*)(pp + 28));

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
                _sum4 = _mm_shuffle_epi32(_sum4, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum6 = _mm_shuffle_epi32(_sum6, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum6);
                __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum6);
                __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum7);
                __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum7);
                __m128i _tmp4 = _mm_unpacklo_epi32(_sum2, _sum4);
                __m128i _tmp5 = _mm_unpackhi_epi32(_sum2, _sum4);
                __m128i _tmp6 = _mm_unpacklo_epi32(_sum3, _sum5);
                __m128i _tmp7 = _mm_unpackhi_epi32(_sum3, _sum5);
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

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale);
            __m128 _f2 = _mm_mul_ps(_mm_cvtepi32_ps(_sum2), _descale);
            __m128 _f3 = _mm_mul_ps(_mm_cvtepi32_ps(_sum3), _descale);
            __m128 _f4 = _mm_mul_ps(_mm_cvtepi32_ps(_sum4), _descale);
            __m128 _f5 = _mm_mul_ps(_mm_cvtepi32_ps(_sum5), _descale);
            __m128 _f6 = _mm_mul_ps(_mm_cvtepi32_ps(_sum6), _descale);
            __m128 _f7 = _mm_mul_ps(_mm_cvtepi32_ps(_sum7), _descale);

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
                    if (c_elempack == 1)
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
                    if (c_elempack == 1)
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
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f4);
                    _mm_storeu_ps(p0 + 8, _f1);
                    _mm_storeu_ps(p0 + 12, _f5);
                    _mm_storeu_ps(p0 + 16, _f2);
                    _mm_storeu_ps(p0 + 20, _f6);
                    _mm_storeu_ps(p0 + 24, _f3);
                    _mm_storeu_ps(p0 + 28, _f7);
                }
#endif // __AVX__
                if (out_elempack == 4)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _MM_TRANSPOSE4_PS(_f4, _f5, _f6, _f7);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
                    _mm_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm_storeu_ps(p0 + out_hstep * 4 + 4, _f5);
                    _mm_storeu_ps(p0 + out_hstep * 4 + 8, _f6);
                    _mm_storeu_ps(p0 + out_hstep * 4 + 12, _f7);
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
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
                    _mm_storeu_ps(p0 + 16, _f4);
                    _mm_storeu_ps(p0 + 20, _f5);
                    _mm_storeu_ps(p0 + 24, _f6);
                    _mm_storeu_ps(p0 + 28, _f7);
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
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));
            __m128i _sum2 = _mm_loadu_si128((const __m128i*)(pp + 8));
            __m128i _sum3 = _mm_loadu_si128((const __m128i*)(pp + 12));

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

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale);
            __m128 _f2 = _mm_mul_ps(_mm_cvtepi32_ps(_sum2), _descale);
            __m128 _f3 = _mm_mul_ps(_mm_cvtepi32_ps(_sum3), _descale);

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
                    if (c_elempack == 1)
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
                if (out_elempack == 4)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
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
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
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
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));

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

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale);

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
                    if (c_elempack == 1)
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
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    p0 += 8;
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

                    p0 += 2;
                }
            }

            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)pp)), _descale);

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
                    if (c_elempack == 1)
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
                    _mm_storeu_ps(p0, _f0);
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    _mm_storeu_ps(sum0, _f0);
                    p0[0] = sum0[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0++;
                }
            }

            pp += 4;
        }
    }
#else
    const int* pp = (const int*)topT;
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
#endif

        float c0;
        float c1;
#if __SSE2__
        __m128 _c0;
        __m128 _c1;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
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
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));
            __m128i _sum2 = _mm_loadu_si128((const __m128i*)(pp + 8));
            __m128i _sum3 = _mm_loadu_si128((const __m128i*)(pp + 12));

            // 00 11 02 13
            // 04 15 06 17
            // 10 01 12 03
            // 14 05 16 07
            _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 3, 0, 1));
            _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 3, 0, 1));

            // 00 11 02 13
            // 04 15 06 17
            // 01 10 03 12
            // 05 14 07 16

            __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum2);
            __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum2);
            __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum3);
            __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum3);

            // 00 01 11 10
            // 02 03 13 12
            // 04 05 15 14
            // 06 07 17 16

            _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
            _sum1 = _mm_unpacklo_epi64(_tmp2, _tmp3);
            _sum2 = _mm_unpackhi_epi64(_tmp0, _tmp1);
            _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);

            // 00 01 02 03
            // 04 05 06 07
            // 11 10 13 12
            // 15 14 17 16
            _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 3, 0, 1));
            _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 3, 0, 1));

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale0);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale0);
            __m128 _f2 = _mm_mul_ps(_mm_cvtepi32_ps(_sum2), _descale1);
            __m128 _f3 = _mm_mul_ps(_mm_cvtepi32_ps(_sum3), _descale1);

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
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
                }
#endif // __AVX__
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 4, _f1);
                    _mm_storeu_ps(p0 + out_hstep * 4 + 4, _f3);
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
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));

            // 00 11 02 13
            // 01 12 03 10
            __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
            __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum1);

            // 00 01 11 12
            // 02 03 13 10
            _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
            _sum1 = _mm_unpackhi_epi64(_tmp1, _tmp0);

            // 00 01 02 03
            // 13 10 11 12
            _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(0, 3, 2, 1));

            // 00 01 02 03
            // 10 11 12 13

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale0);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale1);

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
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
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
#endif

        float c0;
#if __SSE2__
        __m128 _c0;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
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
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        gemm_transB_packed_tile_int8_avx2(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_xop())
    {
        gemm_transB_packed_tile_int8_xop(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

    NCNN_LOGE("gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __SSE2__
#if __AVX2__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // // 0123 4567
                // // 4567 0123
                // __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 6745 2301
                __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(0, 1, 2, 3));

                // 0123 4567
                // 1230 5674
                // 2301 6745
                // 3012 7456
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 1, 0, 3));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA1, _pB0));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA1, _pB1));
                _sum4 = _mm256_add_epi32(_sum4, _mm256_madd_epi16(_pA0, _pB2));
                _sum5 = _mm256_add_epi32(_sum5, _mm256_madd_epi16(_pA0, _pB3));
                _sum6 = _mm256_add_epi32(_sum6, _mm256_madd_epi16(_pA1, _pB2));
                _sum7 = _mm256_add_epi32(_sum7, _mm256_madd_epi16(_pA1, _pB3));

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // // 0123 4567
                // // 4567 0123
                // __m128i _pA0 = _pA;
                // __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 6745 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 1, 2, 3));

                // 0123 4567
                // 1230 5674
                // 2301 6745
                // 3012 7456
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB2 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(2, 3, 0, 1));
                __m128i _pB3 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(2, 1, 0, 3)), _MM_SHUFFLE(2, 1, 0, 3));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1));
                __m256i _s4 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB2));
                __m256i _s5 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB3));
                __m256i _s6 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB2));
                __m256i _s7 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB3));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);
                _sum4 = _mm256_add_epi32(_sum4, _s4);
                _sum5 = _mm256_add_epi32(_sum5, _s5);
                _sum6 = _mm256_add_epi32(_sum6, _s6);
                _sum7 = _mm256_add_epi32(_sum7, _s7);

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castpd_si128(_mm_load1_pd((const double*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567
                // 2301 6745
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 0123
                // 1230 1230
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA1, _pB0));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA1, _pB1));

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01234567
                // 23016745
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 01230123
                // 12301230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567

                // 0101 0101
                // 1010 1010
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 1, 0, 1));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01234567

                // 01010101
                // 10101010
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

                // 0xxx0xxx -> 00000000 11111111
                __m256i _pB0 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(0, 0, 0, 0));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));

                pA += 16;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(pB[0]);

                _pA = _mm_cvtepi8_epi16(_pA);

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB));

                _sum0 = _mm256_add_epi32(_sum0, _s0);

                pA += 8;
                pB += 1;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;
            // NCNN_LOGE("pA %p", pA);
            // if (max_jj == 12)
            // {
            //     NCNN_LOGE("%d %d %d %d %d %d %d %d", pA[0], pA[1], pA[2], pA[3], pA[4], pA[5], pA[6], pA[7]);
            //     NCNN_LOGE("%d %d %d %d %d %d %d %d", pB[0], pB[1], pB[2], pB[3], pB[4], pB[5], pB[6], pB[7]);
            // }

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;
            __m128i _sum4;
            __m128i _sum5;
            __m128i _sum6;
            __m128i _sum7;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
                _sum4 = _mm_setzero_si128();
                _sum5 = _mm_setzero_si128();
                _sum6 = _mm_setzero_si128();
                _sum7 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
                _sum4 = _mm_load_si128((const __m128i*)(outptr + 16));
                _sum5 = _mm_load_si128((const __m128i*)(outptr + 20));
                _sum6 = _mm_load_si128((const __m128i*)(outptr + 24));
                _sum7 = _mm_load_si128((const __m128i*)(outptr + 28));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

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

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maddd_epi16(_pA1, _pB1, _sum3);
                _sum4 = _mm_maddd_epi16(_pA0, _pB2, _sum4);
                _sum5 = _mm_maddd_epi16(_pA0, _pB3, _sum5);
                _sum6 = _mm_maddd_epi16(_pA1, _pB2, _sum6);
                _sum7 = _mm_maddd_epi16(_pA1, _pB3, _sum7);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA1, _pB0));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA1, _pB1));
                _sum4 = _mm_add_epi32(_sum4, _mm_madd_epi16(_pA0, _pB2));
                _sum5 = _mm_add_epi32(_sum5, _mm_madd_epi16(_pA0, _pB3));
                _sum6 = _mm_add_epi32(_sum6, _mm_madd_epi16(_pA1, _pB2));
                _sum7 = _mm_add_epi32(_sum7, _mm_madd_epi16(_pA1, _pB3));
#endif

                // if (max_jj == 12)
                // {
                //     NCNN_LOGE("%d %d %d %d %d %d %d %d", pA[0], pA[1], pA[2], pA[3], pA[4], pA[5], pA[6], pA[7]);
                //     NCNN_LOGE("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", pB[0], pB[1], pB[2], pB[3], pB[4], pB[5], pB[6], pB[7], pB[8], pB[9], pB[10], pB[11], pB[12], pB[13], pB[14], pB[15]);
                // }

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
#endif

                // if (max_jj == 12)
                // {
                //     NCNN_LOGE("%d %d %d %d", pA[0], pA[1], pA[2], pA[3]);
                //     NCNN_LOGE("%d %d %d %d %d %d %d %d", pB[0], pB[1], pB[2], pB[3], pB[4], pB[5], pB[6], pB[7]);
                // }

                pA += 4;
                pB += 8;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            _mm_store_si128((__m128i*)(outptr + 8), _sum2);
            _mm_store_si128((__m128i*)(outptr + 12), _sum3);
            _mm_store_si128((__m128i*)(outptr + 16), _sum4);
            _mm_store_si128((__m128i*)(outptr + 20), _sum5);
            _mm_store_si128((__m128i*)(outptr + 24), _sum6);
            _mm_store_si128((__m128i*)(outptr + 28), _sum7);

            if (max_jj == 12)
            {
                NCNN_LOGE("outptr %d %d %d %d %d %d %d %d", outptr[0], outptr[1], outptr[2], outptr[3], outptr[4], outptr[5], outptr[6], outptr[7]);
            }

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0123

                // 0101
                // 1010
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(2, 3, 0, 1));

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

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB, _sum0);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB));
#endif

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
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
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

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

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

                pA += 2;
                pB += 8;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            _mm_store_si128((__m128i*)(outptr + 8), _sum2);
            _mm_store_si128((__m128i*)(outptr + 12), _sum3);

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

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif

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
                sum01 = outptr[2];
                sum10 = outptr[1];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[2] * pB[0];
                sum1 += pA[3] * pB[1];
                pA += 4;
                pB += 2;
            }
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
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
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
                _sum0 = _mm_loadu_si128((const __m128i*)outptr);
                _sum1 = _mm_loadu_si128((const __m128i*)(outptr + 4));
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pB0 = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pB1 = _mm_unpackhi_epi8(_pB, _extpB);

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif

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

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB, _sum0);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB));
#endif

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[0] * pB[2];
                sum1 += pA[1] * pB[3];
                pA += 2;
                pB += 4;
            }
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
    TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX__
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
    TILE_N = std::max(8, tile_size / 8 * 8);
#else
    TILE_N = std::max(4, tile_size / 4 * 4);
#endif
#else
    TILE_N = std::max(1, tile_size);
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
#elif __AVX__
            TILE_M = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
            TILE_M = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
#endif

#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
            TILE_N = std::max(8, tile_size / 8 * 8);
#else
            TILE_N = std::max(4, tile_size / 4 * 4);
#endif
#else
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
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#endif
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
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#else
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#endif
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

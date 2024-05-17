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

static void pack_A_tile(const Mat& A, Mat& AT, int batch, int max_ii, int max_kk)
{
    const int N = max_kk * batch;

    for (int b = 0; b < batch; b++)
    {
        float* pp = AT.row(b);

        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            const float* p0 = (const float*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[2 * N];
                pp[3] = p0[3 * N];
                pp[4] = p0[4 * N];
                pp[5] = p0[5 * N];
                pp[6] = p0[6 * N];
                pp[7] = p0[7 * N];
                pp[8] = p0[8 * N];
                pp[9] = p0[9 * N];
                pp[10] = p0[10 * N];
                pp[11] = p0[11 * N];
                pp[12] = p0[12 * N];
                pp[13] = p0[13 * N];
                pp[14] = p0[14 * N];
                pp[15] = p0[15 * N];
                p0 += batch;
                pp += 16;
            }
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[2 * N];
                pp[3] = p0[3 * N];
                pp[4] = p0[4 * N];
                pp[5] = p0[5 * N];
                pp[6] = p0[6 * N];
                pp[7] = p0[7 * N];
                p0 += batch;
                pp += 8;
            }
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[2 * N];
                pp[3] = p0[3 * N];
                p0 += batch;
                pp += 4;
            }
        }
#endif // __SSE2__
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + ii * N + b;

            int kk = 0;
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
            const float* p0 = (const float*)A + ii * N + b;

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

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk, int nT)
{
    #pragma omp parallel for num_threads(nT)
    for (int b = 0; b < batch; b++)
    {
        float* pp = BT.row(b);

        int jj = 0;
#if __SSE2__
        for (; jj + 11 < max_jj; jj += 12)
        {
            const float* p0 = B;

            int kk = 0;
#if __AVX__
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                __m512 _r4 = _mm512_load_ps(p0 + 16 * 4);
                __m512 _r5 = _mm512_load_ps(p0 + 16 * 5);
                __m512 _r6 = _mm512_load_ps(p0 + 16 * 6);
                __m512 _r7 = _mm512_load_ps(p0 + 16 * 7);
                __m512 _r8 = _mm512_load_ps(p0 + 16 * 8);
                __m512 _r9 = _mm512_load_ps(p0 + 16 * 9);
                __m512 _ra = _mm512_load_ps(p0 + 16 * 10);
                __m512 _rb = _mm512_load_ps(p0 + 16 * 11);
                transpose16x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                _mm512_storeu_ps(pp, _r0);
                _mm512_storeu_ps(pp + 16, _r1);
                _mm512_storeu_ps(pp + 16 * 2, _r2);
                _mm512_storeu_ps(pp + 16 * 3, _r3);
                _mm512_storeu_ps(pp + 16 * 4, _r4);
                _mm512_storeu_ps(pp + 16 * 5, _r5);
                _mm512_storeu_ps(pp + 16 * 6, _r6);
                _mm512_storeu_ps(pp + 16 * 7, _r7);
                _mm512_storeu_ps(pp + 16 * 8, _r8);
                _mm512_storeu_ps(pp + 16 * 9, _r9);
                _mm512_storeu_ps(pp + 16 * 10, _ra);
                _mm512_storeu_ps(pp + 16 * 11, _rb);
                p0 += max_jj * batch * 16;
                pp += 192;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                __m256 _r4 = _mm256_load_ps(p0 + 8 * 4);
                __m256 _r5 = _mm256_load_ps(p0 + 8 * 5);
                __m256 _r6 = _mm256_load_ps(p0 + 8 * 6);
                __m256 _r7 = _mm256_load_ps(p0 + 8 * 7);
                __m256 _r8 = _mm256_load_ps(p0 + 8 * 8);
                __m256 _r9 = _mm256_load_ps(p0 + 8 * 9);
                __m256 _ra = _mm256_load_ps(p0 + 8 * 10);
                __m256 _rb = _mm256_load_ps(p0 + 8 * 11);
                transpose8x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 8 * 2, _r2);
                _mm256_storeu_ps(pp + 8 * 3, _r3);
                _mm256_storeu_ps(pp + 8 * 4, _r4);
                _mm256_storeu_ps(pp + 8 * 5, _r5);
                _mm256_storeu_ps(pp + 8 * 6, _r6);
                _mm256_storeu_ps(pp + 8 * 7, _r7);
                _mm256_storeu_ps(pp + 8 * 8, _r8);
                _mm256_storeu_ps(pp + 8 * 9, _r9);
                _mm256_storeu_ps(pp + 8 * 10, _ra);
                _mm256_storeu_ps(pp + 8 * 11, _rb);
                p0 += max_jj * batch * 8;
                pp += 96;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __AVX__
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                __m128 _r4 = _mm_load_ps(p0 + 4 * 4);
                __m128 _r5 = _mm_load_ps(p0 + 4 * 5);
                __m128 _r6 = _mm_load_ps(p0 + 4 * 6);
                __m128 _r7 = _mm_load_ps(p0 + 4 * 7);
                __m128 _r8 = _mm_load_ps(p0 + 4 * 8);
                __m128 _r9 = _mm_load_ps(p0 + 4 * 9);
                __m128 _ra = _mm_load_ps(p0 + 4 * 10);
                __m128 _rb = _mm_load_ps(p0 + 4 * 11);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4, _r4);
                _mm_store_ps(pp + 4 * 2, _r8);
                _mm_store_ps(pp + 4 * 3, _r1);
                _mm_store_ps(pp + 4 * 4, _r5);
                _mm_store_ps(pp + 4 * 5, _r9);
                _mm_store_ps(pp + 4 * 6, _r2);
                _mm_store_ps(pp + 4 * 7, _r6);
                _mm_store_ps(pp + 4 * 8, _ra);
                _mm_store_ps(pp + 4 * 9, _r3);
                _mm_store_ps(pp + 4 * 10, _r7);
                _mm_store_ps(pp + 4 * 11, _rb);
                p0 += max_jj * batch * 4;
                pp += 48;
            }
            p0 -= (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[2];
                pp[2] = p0[2 * 2];
                pp[3] = p0[3 * 2];
                pp[4] = p0[4 * 2];
                pp[5] = p0[5 * 2];
                pp[6] = p0[6 * 2];
                pp[7] = p0[7 * 2];
                pp[8] = p0[8 * 2];
                pp[9] = p0[9 * 2];
                pp[10] = p0[10 * 2];
                pp[11] = p0[11 * 2];
                pp[12] = p0[1];
                pp[13] = p0[2 + 1];
                pp[14] = p0[2 * 2 + 1];
                pp[15] = p0[3 * 2 + 1];
                pp[16] = p0[4 * 2 + 1];
                pp[17] = p0[5 * 2 + 1];
                pp[18] = p0[6 * 2 + 1];
                pp[19] = p0[7 * 2 + 1];
                pp[20] = p0[8 * 2 + 1];
                pp[21] = p0[9 * 2 + 1];
                pp[22] = p0[10 * 2 + 1];
                pp[23] = p0[11 * 2 + 1];
                p0 += max_jj * batch * 2;
                pp += 24;
            }
            p0 -= (b * max_jj + jj);
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
                pp[8] = p0[8];
                pp[9] = p0[9];
                pp[10] = p0[10];
                pp[11] = p0[11];
                p0 += max_jj * batch;
                pp += 12;
            }
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* p0 = B;

            int kk = 0;
#if __AVX__
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                __m512 _r4 = _mm512_load_ps(p0 + 16 * 4);
                __m512 _r5 = _mm512_load_ps(p0 + 16 * 5);
                __m512 _r6 = _mm512_load_ps(p0 + 16 * 6);
                __m512 _r7 = _mm512_load_ps(p0 + 16 * 7);
                transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm512_storeu_ps(pp, _r0);
                _mm512_storeu_ps(pp + 16, _r1);
                _mm512_storeu_ps(pp + 16 * 2, _r2);
                _mm512_storeu_ps(pp + 16 * 3, _r3);
                _mm512_storeu_ps(pp + 16 * 4, _r4);
                _mm512_storeu_ps(pp + 16 * 5, _r5);
                _mm512_storeu_ps(pp + 16 * 6, _r6);
                _mm512_storeu_ps(pp + 16 * 7, _r7);
                p0 += max_jj * batch * 16;
                pp += 128;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                __m256 _r4 = _mm256_load_ps(p0 + 8 * 4);
                __m256 _r5 = _mm256_load_ps(p0 + 8 * 5);
                __m256 _r6 = _mm256_load_ps(p0 + 8 * 6);
                __m256 _r7 = _mm256_load_ps(p0 + 8 * 7);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 8 * 2, _r2);
                _mm256_storeu_ps(pp + 8 * 3, _r3);
                _mm256_storeu_ps(pp + 8 * 4, _r4);
                _mm256_storeu_ps(pp + 8 * 5, _r5);
                _mm256_storeu_ps(pp + 8 * 6, _r6);
                _mm256_storeu_ps(pp + 8 * 7, _r7);
                p0 += max_jj * batch * 8;
                pp += 64;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __AVX__
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                __m128 _r4 = _mm_load_ps(p0 + 4 * 4);
                __m128 _r5 = _mm_load_ps(p0 + 4 * 5);
                __m128 _r6 = _mm_load_ps(p0 + 4 * 6);
                __m128 _r7 = _mm_load_ps(p0 + 4 * 7);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4, _r4);
                _mm_store_ps(pp + 8, _r1);
                _mm_store_ps(pp + 12, _r5);
                _mm_store_ps(pp + 16, _r2);
                _mm_store_ps(pp + 20, _r6);
                _mm_store_ps(pp + 24, _r3);
                _mm_store_ps(pp + 28, _r7);
                p0 += max_jj * batch * 4;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[2];
                pp[2] = p0[4];
                pp[3] = p0[6];
                pp[4] = p0[8];
                pp[5] = p0[10];
                pp[6] = p0[12];
                pp[7] = p0[14];
                pp[8] = p0[1];
                pp[9] = p0[3];
                pp[10] = p0[5];
                pp[11] = p0[7];
                pp[12] = p0[9];
                pp[13] = p0[11];
                pp[14] = p0[13];
                pp[15] = p0[15];
                p0 += max_jj * batch * 2;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
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
                p0 += max_jj * batch;
                pp += 8;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* p0 = B;

            int kk = 0;
#if __AVX__
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm512_storeu_ps(pp, _r0);
                _mm512_storeu_ps(pp + 16, _r1);
                _mm512_storeu_ps(pp + 32, _r2);
                _mm512_storeu_ps(pp + 48, _r3);
                p0 += max_jj * batch * 16;
                pp += 64;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 8 * 2, _r2);
                _mm256_storeu_ps(pp + 8 * 3, _r3);
                p0 += max_jj * batch * 8;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __AVX__
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4, _r1);
                _mm_store_ps(pp + 8, _r2);
                _mm_store_ps(pp + 12, _r3);
                p0 += max_jj * batch * 4;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[2];
                pp[2] = p0[4];
                pp[3] = p0[6];
                pp[4] = p0[1];
                pp[5] = p0[3];
                pp[6] = p0[5];
                pp[7] = p0[7];
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
            const float* p0 = B;

            int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16);
                transpose16x2_ps(_r0, _r1);
                _mm512_storeu_ps(pp, _r0);
                _mm512_storeu_ps(pp + 16, _r1);
                p0 += max_jj * batch * 16;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8);
                transpose8x2_ps(_r0, _r1);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                p0 += max_jj * batch * 8;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __AVX__
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4);
                __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                _mm_storeu_ps(pp, _tmp0);
                _mm_storeu_ps(pp + 4, _tmp1);
                p0 += max_jj * batch * 4;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 4;
#endif // __SSE2__
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[2];
                pp[2] = p0[1];
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
            const float* p0 = B;

            int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            p0 += (b * max_jj + jj) * 16;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                _mm512_storeu_ps(pp, _r0);
                p0 += max_jj * batch * 16;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 16;
#endif // __AVX512F__
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                _mm256_storeu_ps(pp, _r0);
                p0 += max_jj * batch * 8;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __AVX__
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                _mm_storeu_ps(pp, _r0);
                p0 += max_jj * batch * 4;
                pp += 4;
            }
            p0 -= (b * max_jj + jj) * 4;
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

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk)
{
    float* outptr = top_blob;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

                __m512 _sum0;
                __m512 _sum1;
                __m512 _sum2;
                __m512 _sum3;
                __m512 _sum4;
                __m512 _sum5;
                __m512 _sum6;
                __m512 _sum7;
                __m512 _sum8;
                __m512 _sum9;
                __m512 _suma;
                __m512 _sumb;

                if (k == 0)
                {
                    _sum0 = _mm512_setzero_ps();
                    _sum1 = _mm512_setzero_ps();
                    _sum2 = _mm512_setzero_ps();
                    _sum3 = _mm512_setzero_ps();
                    _sum4 = _mm512_setzero_ps();
                    _sum5 = _mm512_setzero_ps();
                    _sum6 = _mm512_setzero_ps();
                    _sum7 = _mm512_setzero_ps();
                    _sum8 = _mm512_setzero_ps();
                    _sum9 = _mm512_setzero_ps();
                    _suma = _mm512_setzero_ps();
                    _sumb = _mm512_setzero_ps();
                }
                else
                {
                    _sum0 = _mm512_load_ps(outptr);
                    _sum1 = _mm512_load_ps(outptr + 16);
                    _sum2 = _mm512_load_ps(outptr + 16 * 2);
                    _sum3 = _mm512_load_ps(outptr + 16 * 3);
                    _sum4 = _mm512_load_ps(outptr + 16 * 4);
                    _sum5 = _mm512_load_ps(outptr + 16 * 5);
                    _sum6 = _mm512_load_ps(outptr + 16 * 6);
                    _sum7 = _mm512_load_ps(outptr + 16 * 7);
                    _sum8 = _mm512_load_ps(outptr + 16 * 8);
                    _sum9 = _mm512_load_ps(outptr + 16 * 9);
                    _suma = _mm512_load_ps(outptr + 16 * 10);
                    _sumb = _mm512_load_ps(outptr + 16 * 11);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m512 _pA = _mm512_load_ps(pA);
                    _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);
                    _sum4 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[4]), _sum4);
                    _sum5 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[5]), _sum5);
                    _sum6 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[6]), _sum6);
                    _sum7 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[7]), _sum7);
                    _sum8 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[8]), _sum8);
                    _sum9 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[9]), _sum9);
                    _suma = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[10]), _suma);
                    _sumb = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[11]), _sumb);

                    pA += 16;
                    pB += 12;
                }

                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                _mm512_store_ps(outptr + 16 * 4, _sum4);
                _mm512_store_ps(outptr + 16 * 5, _sum5);
                _mm512_store_ps(outptr + 16 * 6, _sum6);
                _mm512_store_ps(outptr + 16 * 7, _sum7);
                _mm512_store_ps(outptr + 16 * 8, _sum8);
                _mm512_store_ps(outptr + 16 * 9, _sum9);
                _mm512_store_ps(outptr + 16 * 10, _suma);
                _mm512_store_ps(outptr + 16 * 11, _sumb);
                outptr += 16 * 12;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

                __m512 _sum0;
                __m512 _sum1;
                __m512 _sum2;
                __m512 _sum3;
                __m512 _sum4;
                __m512 _sum5;
                __m512 _sum6;
                __m512 _sum7;

                if (k == 0)
                {
                    _sum0 = _mm512_setzero_ps();
                    _sum1 = _mm512_setzero_ps();
                    _sum2 = _mm512_setzero_ps();
                    _sum3 = _mm512_setzero_ps();
                    _sum4 = _mm512_setzero_ps();
                    _sum5 = _mm512_setzero_ps();
                    _sum6 = _mm512_setzero_ps();
                    _sum7 = _mm512_setzero_ps();
                }
                else
                {
                    _sum0 = _mm512_load_ps(outptr);
                    _sum1 = _mm512_load_ps(outptr + 16);
                    _sum2 = _mm512_load_ps(outptr + 16 * 2);
                    _sum3 = _mm512_load_ps(outptr + 16 * 3);
                    _sum4 = _mm512_load_ps(outptr + 16 * 4);
                    _sum5 = _mm512_load_ps(outptr + 16 * 5);
                    _sum6 = _mm512_load_ps(outptr + 16 * 6);
                    _sum7 = _mm512_load_ps(outptr + 16 * 7);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m512 _pA = _mm512_load_ps(pA);
                    _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);
                    _sum4 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[4]), _sum4);
                    _sum5 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[5]), _sum5);
                    _sum6 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[6]), _sum6);
                    _sum7 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[7]), _sum7);

                    pA += 16;
                    pB += 8;
                }

                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                _mm512_store_ps(outptr + 16 * 4, _sum4);
                _mm512_store_ps(outptr + 16 * 5, _sum5);
                _mm512_store_ps(outptr + 16 * 6, _sum6);
                _mm512_store_ps(outptr + 16 * 7, _sum7);
                outptr += 16 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m512 _sum0;
                __m512 _sum1;
                __m512 _sum2;
                __m512 _sum3;

                if (k == 0)
                {
                    _sum0 = _mm512_setzero_ps();
                    _sum1 = _mm512_setzero_ps();
                    _sum2 = _mm512_setzero_ps();
                    _sum3 = _mm512_setzero_ps();
                }
                else
                {
                    _sum0 = _mm512_load_ps(outptr);
                    _sum1 = _mm512_load_ps(outptr + 16);
                    _sum2 = _mm512_load_ps(outptr + 16 * 2);
                    _sum3 = _mm512_load_ps(outptr + 16 * 3);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m512 _pA = _mm512_load_ps(pA);
                    _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);

                    pA += 16;
                    pB += 4;
                }

                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                outptr += 16 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                __m512 _sum0;
                __m512 _sum1;

                if (k == 0)
                {
                    _sum0 = _mm512_setzero_ps();
                    _sum1 = _mm512_setzero_ps();
                }
                else
                {
                    _sum0 = _mm512_load_ps(outptr);
                    _sum1 = _mm512_load_ps(outptr + 16);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m512 _pA = _mm512_load_ps(pA);
                    _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);

                    pA += 16;
                    pB += 2;
                }

                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
                outptr += 16 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const float* pA = pAT;

                __m512 _sum;

                if (k == 0)
                {
                    _sum = _mm512_setzero_ps();
                }
                else
                {
                    _sum = _mm512_load_ps(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m512 _pA = _mm512_load_ps(pA);
                    __m512 _pB = _mm512_set1_ps(pB[0]);
                    _sum = _mm512_fmadd_ps(_pA, _pB, _sum);

                    pA += 16;
                    pB += 1;
                }

                _mm512_store_ps(outptr, _sum);
                outptr += 16;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

                __m256 _sum0;
                __m256 _sum1;
                __m256 _sum2;
                __m256 _sum3;
                __m256 _sum4;
                __m256 _sum5;
                __m256 _sum6;
                __m256 _sum7;
                __m256 _sum8;
                __m256 _sum9;
                __m256 _suma;
                __m256 _sumb;

                if (k == 0)
                {
                    _sum0 = _mm256_setzero_ps();
                    _sum1 = _mm256_setzero_ps();
                    _sum2 = _mm256_setzero_ps();
                    _sum3 = _mm256_setzero_ps();
                    _sum4 = _mm256_setzero_ps();
                    _sum5 = _mm256_setzero_ps();
                    _sum6 = _mm256_setzero_ps();
                    _sum7 = _mm256_setzero_ps();
                    _sum8 = _mm256_setzero_ps();
                    _sum9 = _mm256_setzero_ps();
                    _suma = _mm256_setzero_ps();
                    _sumb = _mm256_setzero_ps();
                }
                else
                {
                    _sum0 = _mm256_load_ps(outptr);
                    _sum1 = _mm256_load_ps(outptr + 8);
                    _sum2 = _mm256_load_ps(outptr + 16);
                    _sum3 = _mm256_load_ps(outptr + 24);
                    _sum4 = _mm256_load_ps(outptr + 32);
                    _sum5 = _mm256_load_ps(outptr + 40);
                    _sum6 = _mm256_load_ps(outptr + 48);
                    _sum7 = _mm256_load_ps(outptr + 56);
                    _sum8 = _mm256_load_ps(outptr + 64);
                    _sum9 = _mm256_load_ps(outptr + 72);
                    _suma = _mm256_load_ps(outptr + 80);
                    _sumb = _mm256_load_ps(outptr + 88);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = _mm256_load_ps(pA);
                    _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[4]), _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[5]), _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[6]), _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[7]), _sum7);
                    _sum8 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[8]), _sum8);
                    _sum9 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[9]), _sum9);
                    _suma = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[10]), _suma);
                    _sumb = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[11]), _sumb);

                    pA += 8;
                    pB += 12;
                }

                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
                _mm256_store_ps(outptr + 8 * 4, _sum4);
                _mm256_store_ps(outptr + 8 * 5, _sum5);
                _mm256_store_ps(outptr + 8 * 6, _sum6);
                _mm256_store_ps(outptr + 8 * 7, _sum7);
                _mm256_store_ps(outptr + 8 * 8, _sum8);
                _mm256_store_ps(outptr + 8 * 9, _sum9);
                _mm256_store_ps(outptr + 8 * 10, _suma);
                _mm256_store_ps(outptr + 8 * 11, _sumb);
                outptr += 8 * 12;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

                __m256 _sum0;
                __m256 _sum1;
                __m256 _sum2;
                __m256 _sum3;
                __m256 _sum4;
                __m256 _sum5;
                __m256 _sum6;
                __m256 _sum7;

                if (k == 0)
                {
                    _sum0 = _mm256_setzero_ps();
                    _sum1 = _mm256_setzero_ps();
                    _sum2 = _mm256_setzero_ps();
                    _sum3 = _mm256_setzero_ps();
                    _sum4 = _mm256_setzero_ps();
                    _sum5 = _mm256_setzero_ps();
                    _sum6 = _mm256_setzero_ps();
                    _sum7 = _mm256_setzero_ps();
                }
                else
                {
                    _sum0 = _mm256_load_ps(outptr);
                    _sum1 = _mm256_load_ps(outptr + 8);
                    _sum2 = _mm256_load_ps(outptr + 16);
                    _sum3 = _mm256_load_ps(outptr + 24);
                    _sum4 = _mm256_load_ps(outptr + 32);
                    _sum5 = _mm256_load_ps(outptr + 40);
                    _sum6 = _mm256_load_ps(outptr + 48);
                    _sum7 = _mm256_load_ps(outptr + 56);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = _mm256_load_ps(pA);
                    _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[4]), _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[5]), _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[6]), _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[7]), _sum7);

                    pA += 8;
                    pB += 8;
                }

                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
                _mm256_store_ps(outptr + 8 * 4, _sum4);
                _mm256_store_ps(outptr + 8 * 5, _sum5);
                _mm256_store_ps(outptr + 8 * 6, _sum6);
                _mm256_store_ps(outptr + 8 * 7, _sum7);
                outptr += 8 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m256 _sum0;
                __m256 _sum1;
                __m256 _sum2;
                __m256 _sum3;

                if (k == 0)
                {
                    _sum0 = _mm256_setzero_ps();
                    _sum1 = _mm256_setzero_ps();
                    _sum2 = _mm256_setzero_ps();
                    _sum3 = _mm256_setzero_ps();
                }
                else
                {
                    _sum0 = _mm256_load_ps(outptr);
                    _sum1 = _mm256_load_ps(outptr + 8);
                    _sum2 = _mm256_load_ps(outptr + 16);
                    _sum3 = _mm256_load_ps(outptr + 24);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = _mm256_load_ps(pA);
                    _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);

                    pA += 8;
                    pB += 4;
                }

                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
                outptr += 8 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                __m256 _sum0;
                __m256 _sum1;

                if (k == 0)
                {
                    _sum0 = _mm256_setzero_ps();
                    _sum1 = _mm256_setzero_ps();
                }
                else
                {
                    _sum0 = _mm256_load_ps(outptr);
                    _sum1 = _mm256_load_ps(outptr + 8);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = _mm256_load_ps(pA);
                    _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);

                    pA += 8;
                    pB += 2;
                }

                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
                outptr += 8 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const float* pA = pAT;

                __m256 _sum;

                if (k == 0)
                {
                    _sum = _mm256_setzero_ps();
                }
                else
                {
                    _sum = _mm256_load_ps(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = _mm256_load_ps(pA);
                    __m256 _pB = _mm256_set1_ps(pB[0]);
                    _sum = _mm256_comp_fmadd_ps(_pA, _pB, _sum);

                    pA += 8;
                    pB += 1;
                }

                _mm256_store_ps(outptr, _sum);
                outptr += 8;
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;
                __m128 _sum3;
                __m128 _sum4;
                __m128 _sum5;
                __m128 _sum6;
                __m128 _sum7;
                __m128 _sum8;
                __m128 _sum9;
                __m128 _suma;
                __m128 _sumb;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                    _sum3 = _mm_setzero_ps();
                    _sum4 = _mm_setzero_ps();
                    _sum5 = _mm_setzero_ps();
                    _sum6 = _mm_setzero_ps();
                    _sum7 = _mm_setzero_ps();
                    _sum8 = _mm_setzero_ps();
                    _sum9 = _mm_setzero_ps();
                    _suma = _mm_setzero_ps();
                    _sumb = _mm_setzero_ps();
                }
                else
                {
                    _sum0 = _mm_load_ps(outptr);
                    _sum1 = _mm_load_ps(outptr + 4);
                    _sum2 = _mm_load_ps(outptr + 8);
                    _sum3 = _mm_load_ps(outptr + 12);
                    _sum4 = _mm_load_ps(outptr + 16);
                    _sum5 = _mm_load_ps(outptr + 20);
                    _sum6 = _mm_load_ps(outptr + 24);
                    _sum7 = _mm_load_ps(outptr + 28);
                    _sum8 = _mm_load_ps(outptr + 32);
                    _sum9 = _mm_load_ps(outptr + 36);
                    _suma = _mm_load_ps(outptr + 40);
                    _sumb = _mm_load_ps(outptr + 44);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_load_ps(pA);
                    _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[4]), _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[5]), _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[6]), _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[7]), _sum7);
                    _sum8 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[8]), _sum8);
                    _sum9 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[9]), _sum9);
                    _suma = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[10]), _suma);
                    _sumb = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[11]), _sumb);

                    pA += 4;
                    pB += 12;
                }

                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
                _mm_store_ps(outptr + 4 * 4, _sum4);
                _mm_store_ps(outptr + 4 * 5, _sum5);
                _mm_store_ps(outptr + 4 * 6, _sum6);
                _mm_store_ps(outptr + 4 * 7, _sum7);
                _mm_store_ps(outptr + 4 * 8, _sum8);
                _mm_store_ps(outptr + 4 * 9, _sum9);
                _mm_store_ps(outptr + 4 * 10, _suma);
                _mm_store_ps(outptr + 4 * 11, _sumb);
                outptr += 4 * 12;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;
                __m128 _sum3;
                __m128 _sum4;
                __m128 _sum5;
                __m128 _sum6;
                __m128 _sum7;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                    _sum3 = _mm_setzero_ps();
                    _sum4 = _mm_setzero_ps();
                    _sum5 = _mm_setzero_ps();
                    _sum6 = _mm_setzero_ps();
                    _sum7 = _mm_setzero_ps();
                }
                else
                {
                    _sum0 = _mm_load_ps(outptr);
                    _sum1 = _mm_load_ps(outptr + 4);
                    _sum2 = _mm_load_ps(outptr + 8);
                    _sum3 = _mm_load_ps(outptr + 12);
                    _sum4 = _mm_load_ps(outptr + 16);
                    _sum5 = _mm_load_ps(outptr + 20);
                    _sum6 = _mm_load_ps(outptr + 24);
                    _sum7 = _mm_load_ps(outptr + 28);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_load_ps(pA);
                    _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[4]), _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[5]), _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[6]), _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[7]), _sum7);

                    pA += 4;
                    pB += 8;
                }

                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
                _mm_store_ps(outptr + 4 * 4, _sum4);
                _mm_store_ps(outptr + 4 * 5, _sum5);
                _mm_store_ps(outptr + 4 * 6, _sum6);
                _mm_store_ps(outptr + 4 * 7, _sum7);
                outptr += 4 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;
                __m128 _sum3;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                    _sum3 = _mm_setzero_ps();
                }
                else
                {
                    _sum0 = _mm_load_ps(outptr);
                    _sum1 = _mm_load_ps(outptr + 4);
                    _sum2 = _mm_load_ps(outptr + 8);
                    _sum3 = _mm_load_ps(outptr + 12);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_load_ps(pA);
                    _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);

                    pA += 4;
                    pB += 4;
                }

                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
                outptr += 4 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                }
                else
                {
                    _sum0 = _mm_load_ps(outptr);
                    _sum1 = _mm_load_ps(outptr + 4);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_load_ps(pA);
                    _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);

                    pA += 4;
                    pB += 2;
                }

                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
                outptr += 4 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const float* pA = pAT;

                __m128 _sum;

                if (k == 0)
                {
                    _sum = _mm_setzero_ps();
                }
                else
                {
                    _sum = _mm_load_ps(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_load_ps(pA);
                    _sum = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum);

                    pA += 4;
                    pB += 1;
                }

                _mm_store_ps(outptr, _sum);
                outptr += 4;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
#if __SSE2__
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;
                __m128 _sum3;
                __m128 _sum4;
                __m128 _sum5;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                    _sum3 = _mm_setzero_ps();
                    _sum4 = _mm_setzero_ps();
                    _sum5 = _mm_setzero_ps();
                }
                else
                {
                    __m128 _tmp0 = _mm_loadu_ps(outptr);
                    __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                    __m128 _tmp2 = _mm_loadu_ps(outptr + 8);
                    __m128 _tmp3 = _mm_loadu_ps(outptr + 12);
                    __m128 _tmp4 = _mm_loadu_ps(outptr + 16);
                    __m128 _tmp5 = _mm_loadu_ps(outptr + 20);
                    _sum0 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum5 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA0 = _mm_set1_ps(pA[0]);
                    __m128 _pA1 = _mm_set1_ps(pA[1]);
                    __m128 _pB0 = _mm_load_ps(pB);
                    __m128 _pB1 = _mm_load_ps(pB + 4);
                    __m128 _pB2 = _mm_load_ps(pB + 8);
                    _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_pA0, _pB2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_pA1, _pB2, _sum5);
                    pA += 2;
                    pB += 12;
                }

                __m128 _tmp0 = _mm_unpacklo_ps(_sum0, _sum3);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum0, _sum3);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum1, _sum4);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum1, _sum4);
                __m128 _tmp4 = _mm_unpacklo_ps(_sum2, _sum5);
                __m128 _tmp5 = _mm_unpackhi_ps(_sum2, _sum5);
                _mm_storeu_ps(outptr, _tmp0);
                _mm_storeu_ps(outptr + 4, _tmp1);
                _mm_storeu_ps(outptr + 8, _tmp2);
                _mm_storeu_ps(outptr + 12, _tmp3);
                _mm_storeu_ps(outptr + 16, _tmp4);
                _mm_storeu_ps(outptr + 20, _tmp5);
                outptr += 2 * 12;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;
                __m128 _sum3;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                    _sum3 = _mm_setzero_ps();
                }
                else
                {
                    __m128 _tmp0 = _mm_loadu_ps(outptr);
                    __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                    __m128 _tmp2 = _mm_loadu_ps(outptr + 8);
                    __m128 _tmp3 = _mm_loadu_ps(outptr + 12);
                    _sum0 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum3 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA0 = _mm_set1_ps(pA[0]);
                    __m128 _pA1 = _mm_set1_ps(pA[1]);
                    __m128 _pB0 = _mm_load_ps(pB);
                    __m128 _pB1 = _mm_load_ps(pB + 4);
                    _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum3);
                    pA += 2;
                    pB += 8;
                }

                __m128 _tmp0 = _mm_unpacklo_ps(_sum0, _sum2);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum0, _sum2);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum1, _sum3);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum1, _sum3);
                _mm_storeu_ps(outptr, _tmp0);
                _mm_storeu_ps(outptr + 4, _tmp1);
                _mm_storeu_ps(outptr + 8, _tmp2);
                _mm_storeu_ps(outptr + 12, _tmp3);
                outptr += 2 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                }
                else
                {
                    __m128 _tmp0 = _mm_loadu_ps(outptr);
                    __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                    _sum0 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pB = _mm_load_ps(pB);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[0]), _pB, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[1]), _pB, _sum1);
                    pA += 2;
                    pB += 4;
                }

                __m128 _tmp0 = _mm_unpacklo_ps(_sum0, _sum1);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum0, _sum1);
                _mm_storeu_ps(outptr, _tmp0);
                _mm_storeu_ps(outptr + 4, _tmp1);
                outptr += 2 * 4;
            }
#endif // __SSE2__
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                float sum00 = 0.f;
                float sum01 = 0.f;
                float sum10 = 0.f;
                float sum11 = 0.f;

                if (k == 0)
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
                }
                else
                {
                    sum00 = outptr[0];
                    sum01 = outptr[1];
                    sum10 = outptr[2];
                    sum11 = outptr[3];
                }

                int kk = 0;
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
                const float* pA = pAT;

                float sum0 = 0.f;
                float sum1 = 0.f;

                if (k == 0)
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
                else
                {
                    sum0 = outptr[0];
                    sum1 = outptr[1];
                }

                int kk = 0;
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
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
#if __SSE2__
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                }
                else
                {
                    _sum0 = _mm_loadu_ps(outptr);
                    _sum1 = _mm_loadu_ps(outptr + 4);
                    _sum2 = _mm_loadu_ps(outptr + 8);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_set1_ps(pA[0]);
                    __m128 _pB0 = _mm_load_ps(pB);
                    __m128 _pB1 = _mm_load_ps(pB + 4);
                    __m128 _pB2 = _mm_load_ps(pB + 8);
                    _sum0 = _mm_comp_fmadd_ps(_pA, _pB0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_pA, _pB1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_pA, _pB2, _sum2);
                    pA += 1;
                    pB += 12;
                }

                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
                _mm_storeu_ps(outptr + 8, _sum2);
                outptr += 12;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;

                if (k == 0)
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                }
                else
                {
                    _sum0 = _mm_loadu_ps(outptr);
                    _sum1 = _mm_loadu_ps(outptr + 4);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_set1_ps(pA[0]);
                    __m128 _pB0 = _mm_load_ps(pB);
                    __m128 _pB1 = _mm_load_ps(pB + 4);
                    _sum0 = _mm_comp_fmadd_ps(_pA, _pB0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_pA, _pB1, _sum1);
                    pA += 1;
                    pB += 8;
                }

                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
                outptr += 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m128 _sum;

                if (k == 0)
                {
                    _sum = _mm_setzero_ps();
                }
                else
                {
                    _sum = _mm_loadu_ps(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_set1_ps(pA[0]);
                    __m128 _pB = _mm_load_ps(pB);
                    _sum = _mm_comp_fmadd_ps(_pA, _pB, _sum);
                    pA += 1;
                    pB += 4;
                }

                _mm_storeu_ps(outptr, _sum);
                outptr += 4;
            }
#endif // __SSE2__
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                float sum0 = 0.f;
                float sum1 = 0.f;

                if (k == 0)
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
                else
                {
                    sum0 = outptr[0];
                    sum1 = outptr[1];
                }

                int kk = 0;
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
                const float* pA = pAT;

                float sum = 0.f;

                if (k == 0)
                {
                    sum = 0.f;
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

static void get_optimal_tile_mnk(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve M
    {
        int tile_size = (int)sqrt((float)l2_cache_size / sizeof(float) / 3);

#if __AVX512F__
        TILE_M = std::max(16, tile_size / 16 * 16);
#elif __AVX__
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

    // solve K
    {
        int tile_size = (int)(sqrt((float)l2_cache_size / sizeof(float)) - TILE_M);

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

    if (N > 0)
    {
        int tile_size = (int)(((float)l2_cache_size / sizeof(float) - TILE_M * TILE_K) / (TILE_M + TILE_K));

#if __AVX512F__
        TILE_N = std::max(4, tile_size / 4 * 4);
#elif __AVX__
        TILE_N = std::max(4, tile_size / 4 * 4);
#elif __SSE2__
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __AVX512F__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __AVX__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __SSE2__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static inline void conv3x3s1_winograd23_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const float ktm[4][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {1.0f / 2, 1.0f / 2, 1.0f / 2},
    //     {1.0f / 2, -1.0f / 2, 1.0f / 2},
    //     {0.0f, 0.0f, 1.0f}
    // };

    float* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            float tmp[4][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = r0 * 0.5f + r1 * 0.5f + r2 * 0.5f;
                tmp[2][m] = r0 * 0.5f - r1 * 0.5f + r2 * 0.5f;
                tmp[3][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 4; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = r0 * 0.5f + r1 * 0.5f + r2 * 0.5f;
                float z2 = r0 * 0.5f - r1 * 0.5f + r2 * 0.5f;
                float z3 = r2;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp += 4;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd23_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd23_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w - 1) / 2;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __SSE2__
#if __AVX__
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
        float tmp[4][4][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m512 _r0 = _mm512_setzero_ps();
                __m512 _r1 = _mm512_setzero_ps();
                __m512 _r2 = _mm512_setzero_ps();
                __m512 _r3 = _mm512_setzero_ps();

                if (ti * 2 + m < h)
                {
                    if (elempack == 16)
                    {
                        _r0 = _mm512_load_ps(r0);
                        if (tj * 2 + 1 < w) _r1 = _mm512_load_ps(r0 + 16);
                        if (tj * 2 + 2 < w) _r2 = _mm512_load_ps(r0 + 32);
                        if (tj * 2 + 3 < w) _r3 = _mm512_load_ps(r0 + 48);
                    }
                    if (elempack == 8)
                    {
                        const float* r1 = r0 + N;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0)), _mm256_load_ps(r1), 1);
                        if (tj * 2 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 8)), _mm256_load_ps(r1 + 8), 1);
                        if (tj * 2 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 16)), _mm256_load_ps(r1 + 16), 1);
                        if (tj * 2 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 24)), _mm256_load_ps(r1 + 24), 1);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2)), _mm_load_ps(r3), 1), 1);
                        if (tj * 2 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 4)), _mm_load_ps(r3 + 4), 1), 1);
                        if (tj * 2 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 8)), _mm_load_ps(r3 + 8), 1), 1);
                        if (tj * 2 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 12)), _mm_load_ps(r3 + 12), 1), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;
                        const float* r8 = r0 + N * 8;
                        const float* r9 = r0 + N * 9;
                        const float* ra = r0 + N * 10;
                        const float* rb = r0 + N * 11;
                        const float* rc = r0 + N * 12;
                        const float* rd = r0 + N * 13;
                        const float* re = r0 + N * 14;
                        const float* rf = r0 + N * 15;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);
                        __m128 _t4 = _mm_loadu_ps(r4);
                        __m128 _t5 = _mm_loadu_ps(r5);
                        __m128 _t6 = _mm_loadu_ps(r6);
                        __m128 _t7 = _mm_loadu_ps(r7);
                        __m128 _t8 = _mm_loadu_ps(r8);
                        __m128 _t9 = _mm_loadu_ps(r9);
                        __m128 _ta = _mm_loadu_ps(ra);
                        __m128 _tb = _mm_loadu_ps(rb);
                        __m128 _tc = _mm_loadu_ps(rc);
                        __m128 _td = _mm_loadu_ps(rd);
                        __m128 _te = _mm_loadu_ps(re);
                        __m128 _tf = _mm_loadu_ps(rf);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);
                        _MM_TRANSPOSE4_PS(_t4, _t5, _t6, _t7);
                        _MM_TRANSPOSE4_PS(_t8, _t9, _ta, _tb);
                        _MM_TRANSPOSE4_PS(_tc, _td, _te, _tf);

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t0), _t4, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_t8), _tc, 1), 1);
                        if (tj * 2 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t1), _t5, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_t9), _td, 1), 1);
                        if (tj * 2 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t2), _t6, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_ta), _te, 1), 1);
                        if (tj * 2 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t3), _t7, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_tb), _tf, 1), 1);
                    }
                }

                __m512 _tmp0 = _mm512_sub_ps(_r0, _r2);
                __m512 _tmp1 = _mm512_add_ps(_r1, _r2);
                __m512 _tmp2 = _mm512_sub_ps(_r2, _r1);
                __m512 _tmp3 = _mm512_sub_ps(_r3, _r1);

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);
                _mm512_store_ps(tmp[2][m], _tmp2);
                _mm512_store_ps(tmp[3][m], _tmp3);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 16;
            float* p1 = p0 + max_jj * 16;
            float* p2 = p0 + max_jj * 16 * 2;
            float* p3 = p0 + max_jj * 16 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m512 _r0 = _mm512_load_ps(tmp[m][0]);
                __m512 _r1 = _mm512_load_ps(tmp[m][1]);
                __m512 _r2 = _mm512_load_ps(tmp[m][2]);
                __m512 _r3 = _mm512_load_ps(tmp[m][3]);

                __m512 _tmp0 = _mm512_sub_ps(_r0, _r2);
                __m512 _tmp1 = _mm512_add_ps(_r1, _r2);
                __m512 _tmp2 = _mm512_sub_ps(_r2, _r1);
                __m512 _tmp3 = _mm512_sub_ps(_r3, _r1);

                _mm512_store_ps(p0, _tmp0);
                _mm512_store_ps(p1, _tmp1);
                _mm512_store_ps(p2, _tmp2);
                _mm512_store_ps(p3, _tmp3);

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
        float tmp[4][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m256 _r0 = _mm256_setzero_ps();
                __m256 _r1 = _mm256_setzero_ps();
                __m256 _r2 = _mm256_setzero_ps();
                __m256 _r3 = _mm256_setzero_ps();

                if (ti * 2 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = _mm256_load_ps(r0);
                        if (tj * 2 + 1 < w) _r1 = _mm256_load_ps(r0 + 8);
                        if (tj * 2 + 2 < w) _r2 = _mm256_load_ps(r0 + 16);
                        if (tj * 2 + 3 < w) _r3 = _mm256_load_ps(r0 + 24);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1);
                        if (tj * 2 + 1 < w) _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1);
                        if (tj * 2 + 2 < w) _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1);
                        if (tj * 2 + 3 < w) _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);
                        __m128 _t4 = _mm_loadu_ps(r4);
                        __m128 _t5 = _mm_loadu_ps(r5);
                        __m128 _t6 = _mm_loadu_ps(r6);
                        __m128 _t7 = _mm_loadu_ps(r7);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);
                        _MM_TRANSPOSE4_PS(_t4, _t5, _t6, _t7);

                        _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t0), _t4, 1);
                        if (tj * 2 + 1 < w) _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t1), _t5, 1);
                        if (tj * 2 + 2 < w) _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t2), _t6, 1);
                        if (tj * 2 + 3 < w) _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t3), _t7, 1);
                    }
                }

                __m256 _tmp0 = _mm256_sub_ps(_r0, _r2);
                __m256 _tmp1 = _mm256_add_ps(_r1, _r2);
                __m256 _tmp2 = _mm256_sub_ps(_r2, _r1);
                __m256 _tmp3 = _mm256_sub_ps(_r3, _r1);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                // old gcc breaks stack variable alignement
                // ref https://gcc.gnu.org/bugzilla/show_bug.cgi?id=16660
                _mm256_storeu_ps(tmp[0][m], _tmp0);
                _mm256_storeu_ps(tmp[1][m], _tmp1);
                _mm256_storeu_ps(tmp[2][m], _tmp2);
                _mm256_storeu_ps(tmp[3][m], _tmp3);
#else
                _mm256_store_ps(tmp[0][m], _tmp0);
                _mm256_store_ps(tmp[1][m], _tmp1);
                _mm256_store_ps(tmp[2][m], _tmp2);
                _mm256_store_ps(tmp[3][m], _tmp3);
#endif

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m256 _r0 = _mm256_loadu_ps(tmp[m][0]);
                __m256 _r1 = _mm256_loadu_ps(tmp[m][1]);
                __m256 _r2 = _mm256_loadu_ps(tmp[m][2]);
                __m256 _r3 = _mm256_loadu_ps(tmp[m][3]);
#else
                __m256 _r0 = _mm256_load_ps(tmp[m][0]);
                __m256 _r1 = _mm256_load_ps(tmp[m][1]);
                __m256 _r2 = _mm256_load_ps(tmp[m][2]);
                __m256 _r3 = _mm256_load_ps(tmp[m][3]);
#endif

                __m256 _tmp0 = _mm256_sub_ps(_r0, _r2);
                __m256 _tmp1 = _mm256_add_ps(_r1, _r2);
                __m256 _tmp2 = _mm256_sub_ps(_r2, _r1);
                __m256 _tmp3 = _mm256_sub_ps(_r3, _r1);

                _mm256_store_ps(p0, _tmp0);
                _mm256_store_ps(p1, _tmp1);
                _mm256_store_ps(p2, _tmp2);
                _mm256_store_ps(p3, _tmp3);

                p0 += max_jj * 4 * 8;
                p1 += max_jj * 4 * 8;
                p2 += max_jj * 4 * 8;
                p3 += max_jj * 4 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
#else // __AVX__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __AVX__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m128 _r0 = _mm_setzero_ps();
                __m128 _r1 = _mm_setzero_ps();
                __m128 _r2 = _mm_setzero_ps();
                __m128 _r3 = _mm_setzero_ps();

                if (ti * 2 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = _mm_load_ps(r0);
                        if (tj * 2 + 1 < w) _r1 = _mm_load_ps(r0 + 4);
                        if (tj * 2 + 2 < w) _r2 = _mm_load_ps(r0 + 8);
                        if (tj * 2 + 3 < w) _r3 = _mm_load_ps(r0 + 12);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 2 + 1 < w) _r1 = _t1;
                        if (tj * 2 + 2 < w) _r2 = _t2;
                        if (tj * 2 + 3 < w) _r3 = _t3;
                    }
                }

                __m128 _tmp0 = _mm_sub_ps(_r0, _r2);
                __m128 _tmp1 = _mm_add_ps(_r1, _r2);
                __m128 _tmp2 = _mm_sub_ps(_r2, _r1);
                __m128 _tmp3 = _mm_sub_ps(_r3, _r1);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_ps(tmp[0][m], _tmp0);
                _mm_storeu_ps(tmp[1][m], _tmp1);
                _mm_storeu_ps(tmp[2][m], _tmp2);
                _mm_storeu_ps(tmp[3][m], _tmp3);
#else
                _mm_store_ps(tmp[0][m], _tmp0);
                _mm_store_ps(tmp[1][m], _tmp1);
                _mm_store_ps(tmp[2][m], _tmp2);
                _mm_store_ps(tmp[3][m], _tmp3);
#endif

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128 _r0 = _mm_loadu_ps(tmp[m][0]);
                __m128 _r1 = _mm_loadu_ps(tmp[m][1]);
                __m128 _r2 = _mm_loadu_ps(tmp[m][2]);
                __m128 _r3 = _mm_loadu_ps(tmp[m][3]);
#else
                __m128 _r0 = _mm_load_ps(tmp[m][0]);
                __m128 _r1 = _mm_load_ps(tmp[m][1]);
                __m128 _r2 = _mm_load_ps(tmp[m][2]);
                __m128 _r3 = _mm_load_ps(tmp[m][3]);
#endif

                __m128 _tmp0 = _mm_sub_ps(_r0, _r2);
                __m128 _tmp1 = _mm_add_ps(_r1, _r2);
                __m128 _tmp2 = _mm_sub_ps(_r2, _r1);
                __m128 _tmp3 = _mm_sub_ps(_r3, _r1);

                _mm_store_ps(p0, _tmp0);
                _mm_store_ps(p1, _tmp1);
                _mm_store_ps(p2, _tmp2);
                _mm_store_ps(p3, _tmp3);

                p0 += max_jj * 4 * 4;
                p1 += max_jj * 4 * 4;
                p2 += max_jj * 4 * 4;
                p3 += max_jj * 4 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __SSE2__
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __SSE2__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        float tmp[4][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

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

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];

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
        float tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;

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

            float* p0 = (float*)B + kk * max_jj * 16 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

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

static inline void conv3x3s1_winograd23_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 1) / 2;

    const float* biasptr = bias;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        __m512 _bias0 = biasptr ? _mm512_loadu_ps(biasptr + i + ii) : _mm512_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[2][4][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 16;
            const float* r1 = r0 + max_jj * 16;
            const float* r2 = r0 + max_jj * 16 * 2;
            const float* r3 = r0 + max_jj * 16 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m512 _r0 = _mm512_load_ps(r0);
                __m512 _r1 = _mm512_load_ps(r1);
                __m512 _r2 = _mm512_load_ps(r2);
                __m512 _r3 = _mm512_load_ps(r3);

                __m512 _tmp0 = _mm512_add_ps(_mm512_add_ps(_r0, _r1), _r2);
                __m512 _tmp1 = _mm512_add_ps(_mm512_sub_ps(_r1, _r2), _r3);

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);

                r0 += max_jj * 4 * 16;
                r1 += max_jj * 4 * 16;
                r2 += max_jj * 4 * 16;
                r3 += max_jj * 4 * 16;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __m512 _r0 = _mm512_load_ps(tmp[m][0]);
                __m512 _r1 = _mm512_load_ps(tmp[m][1]);
                __m512 _r2 = _mm512_load_ps(tmp[m][2]);
                __m512 _r3 = _mm512_load_ps(tmp[m][3]);

                __m512 _tmp0 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_add_ps(_r0, _r1), _r2));
                __m512 _tmp1 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_sub_ps(_r1, _r2), _r3));

                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _tmp0);
                    if (tj * 2 + 1 < outw)
                    {
                        _mm512_store_ps(outptr0 + 16, _tmp1);
                    }
                }
                if (out_elempack == 8)
                {
                    float* outptr1 = outptr0 + N;

                    _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_tmp0, 0));
                    _mm256_store_ps(outptr1, _mm512_extractf32x8_ps(_tmp0, 1));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm256_store_ps(outptr0 + 8, _mm512_extractf32x8_ps(_tmp1, 0));
                        _mm256_store_ps(outptr1 + 8, _mm512_extractf32x8_ps(_tmp1, 1));
                    }
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_tmp0, 0));
                    _mm_store_ps(outptr1, _mm512_extractf32x4_ps(_tmp0, 1));
                    _mm_store_ps(outptr2, _mm512_extractf32x4_ps(_tmp0, 2));
                    _mm_store_ps(outptr3, _mm512_extractf32x4_ps(_tmp0, 3));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm_store_ps(outptr0 + 4, _mm512_extractf32x4_ps(_tmp1, 0));
                        _mm_store_ps(outptr1 + 4, _mm512_extractf32x4_ps(_tmp1, 1));
                        _mm_store_ps(outptr2 + 4, _mm512_extractf32x4_ps(_tmp1, 2));
                        _mm_store_ps(outptr3 + 4, _mm512_extractf32x4_ps(_tmp1, 3));
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[16];
                    float tmp1[16];
                    _mm512_storeu_ps(tmp0, _tmp0);
                    _mm512_storeu_ps(tmp1, _tmp1);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;
                    float* outptr8 = outptr0 + N * 8;
                    float* outptr9 = outptr0 + N * 9;
                    float* outptra = outptr0 + N * 10;
                    float* outptrb = outptr0 + N * 11;
                    float* outptrc = outptr0 + N * 12;
                    float* outptrd = outptr0 + N * 13;
                    float* outptre = outptr0 + N * 14;
                    float* outptrf = outptr0 + N * 15;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    outptr8[0] = tmp0[8];
                    outptr9[0] = tmp0[9];
                    outptra[0] = tmp0[10];
                    outptrb[0] = tmp0[11];
                    outptrc[0] = tmp0[12];
                    outptrd[0] = tmp0[13];
                    outptre[0] = tmp0[14];
                    outptrf[0] = tmp0[15];

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
                        outptr8[1] = tmp1[8];
                        outptr9[1] = tmp1[9];
                        outptra[1] = tmp1[10];
                        outptrb[1] = tmp1[11];
                        outptrc[1] = tmp1[12];
                        outptrd[1] = tmp1[13];
                        outptre[1] = tmp1[14];
                        outptrf[1] = tmp1[15];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _bias0 = biasptr ? _mm256_loadu_ps(biasptr + i + ii) : _mm256_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(32))
#else
        __attribute__((aligned(32)))
#endif
        float tmp[2][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m256 _r0 = _mm256_load_ps(r0);
                __m256 _r1 = _mm256_load_ps(r1);
                __m256 _r2 = _mm256_load_ps(r2);
                __m256 _r3 = _mm256_load_ps(r3);

                __m256 _tmp0 = _mm256_add_ps(_mm256_add_ps(_r0, _r1), _r2);
                __m256 _tmp1 = _mm256_add_ps(_mm256_sub_ps(_r1, _r2), _r3);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm256_storeu_ps(tmp[0][m], _tmp0);
                _mm256_storeu_ps(tmp[1][m], _tmp1);
#else
                _mm256_store_ps(tmp[0][m], _tmp0);
                _mm256_store_ps(tmp[1][m], _tmp1);
#endif

                r0 += max_jj * 4 * 8;
                r1 += max_jj * 4 * 8;
                r2 += max_jj * 4 * 8;
                r3 += max_jj * 4 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m256 _r0 = _mm256_loadu_ps(tmp[m][0]);
                __m256 _r1 = _mm256_loadu_ps(tmp[m][1]);
                __m256 _r2 = _mm256_loadu_ps(tmp[m][2]);
                __m256 _r3 = _mm256_loadu_ps(tmp[m][3]);
#else
                __m256 _r0 = _mm256_load_ps(tmp[m][0]);
                __m256 _r1 = _mm256_load_ps(tmp[m][1]);
                __m256 _r2 = _mm256_load_ps(tmp[m][2]);
                __m256 _r3 = _mm256_load_ps(tmp[m][3]);
#endif

                __m256 _tmp0 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_r0, _r1), _r2));
                __m256 _tmp1 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_sub_ps(_r1, _r2), _r3));

                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _tmp0);
                    if (tj * 2 + 1 < outw)
                    {
                        _mm256_store_ps(outptr0 + 8, _tmp1);
                    }
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    _mm_store_ps(outptr0, _mm256_extractf128_ps(_tmp0, 0));
                    _mm_store_ps(outptr1, _mm256_extractf128_ps(_tmp0, 1));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm_store_ps(outptr0 + 4, _mm256_extractf128_ps(_tmp1, 0));
                        _mm_store_ps(outptr1 + 4, _mm256_extractf128_ps(_tmp1, 1));
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    _mm256_storeu_ps(tmp0, _tmp0);
                    _mm256_storeu_ps(tmp1, _tmp1);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

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
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _bias0 = biasptr ? _mm_loadu_ps(biasptr + i + ii) : _mm_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[2][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m128 _r0 = _mm_load_ps(r0);
                __m128 _r1 = _mm_load_ps(r1);
                __m128 _r2 = _mm_load_ps(r2);
                __m128 _r3 = _mm_load_ps(r3);

                __m128 _tmp0 = _mm_add_ps(_mm_add_ps(_r0, _r1), _r2);
                __m128 _tmp1 = _mm_add_ps(_mm_sub_ps(_r1, _r2), _r3);

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_ps(tmp[0][m], _tmp0);
                _mm_storeu_ps(tmp[1][m], _tmp1);
#else
                _mm_store_ps(tmp[0][m], _tmp0);
                _mm_store_ps(tmp[1][m], _tmp1);
#endif

                r0 += max_jj * 4 * 4;
                r1 += max_jj * 4 * 4;
                r2 += max_jj * 4 * 4;
                r3 += max_jj * 4 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128 _r0 = _mm_loadu_ps(tmp[m][0]);
                __m128 _r1 = _mm_loadu_ps(tmp[m][1]);
                __m128 _r2 = _mm_loadu_ps(tmp[m][2]);
                __m128 _r3 = _mm_loadu_ps(tmp[m][3]);
#else
                __m128 _r0 = _mm_load_ps(tmp[m][0]);
                __m128 _r1 = _mm_load_ps(tmp[m][1]);
                __m128 _r2 = _mm_load_ps(tmp[m][2]);
                __m128 _r3 = _mm_load_ps(tmp[m][3]);
#endif

                __m128 _tmp0 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_r0, _r1), _r2));
                __m128 _tmp1 = _mm_add_ps(_bias0, _mm_add_ps(_mm_sub_ps(_r1, _r2), _r3));

                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _tmp0);
                    if (tj * 2 + 1 < outw) _mm_store_ps(outptr0 + 4, _tmp1);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    _mm_storeu_ps(tmp0, _tmp0);
                    _mm_storeu_ps(tmp1, _tmp1);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

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
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

        float tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;

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

            float* outptr0 = top_blob.channel(i + ii).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];

                float tmp00 = bias0 + r00 + r10 + r20;
                float tmp01 = bias1 + r01 + r11 + r21;
                float tmp10 = bias0 + r10 - r20 + r30;
                float tmp11 = bias1 + r11 - r21 + r31;

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

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
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

                float tmp0 = bias0 + r0 + r1 + r2;
                float tmp1 = bias0 + r1 - r2 + r3;

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

static void conv3x3s1_winograd23(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
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

    // NCNN_LOGE("conv3x3s1_winograd23 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd23_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd23_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
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

                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    float* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            const float sq2 = 1.41421356237f;
            // const float ktm[6][3] = {
            //     {1.0f, 0.0f, 0.0f},
            //     {-2.0f / 3, -sq2 / 3, -1.0f / 3},
            //     {-2.0f / 3, sq2 / 3, -1.0f / 3},
            //     {1.0f / 6, sq2 / 6, 1.0f / 3},
            //     {1.0f / 6, -sq2 / 6, 1.0f / 3},
            //     {0.0f, 0.0f, 1.0f}
            // };
            const float ktm0 = 2.0f / 3;
            const float ktm1 = sq2 / 3;
            const float ktm2 = 1.0f / 3;
            const float ktm3 = 1.0f / 6;
            const float ktm4 = sq2 / 6;

            float tmp[6][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = -r0 * ktm0 - r1 * ktm1 - r2 * ktm2;
                tmp[2][m] = -r0 * ktm0 + r1 * ktm1 - r2 * ktm2;
                tmp[3][m] = r0 * ktm3 + r1 * ktm4 + r2 * ktm2;
                tmp[4][m] = r0 * ktm3 - r1 * ktm4 + r2 * ktm2;
                tmp[5][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = -r0 * ktm0 - r1 * ktm1 - r2 * ktm2;
                float z2 = -r0 * ktm0 + r1 * ktm1 - r2 * ktm2;
                float z3 = r0 * ktm3 + r1 * ktm4 + r2 * ktm2;
                float z4 = r0 * ktm3 - r1 * ktm4 + r2 * ktm2;
                float z5 = r2;

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

static void conv3x3s1_winograd43_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd43_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    const float sq2 = 1.41421356237;
    const float sq2_d2 = 1.41421356237 / 2;

    // const float itm[6][6] = {
    //     {1.0f,  0.0f,  -2.5f,  0.0f,  1.0f, 0.0f},
    //     {0.0f, -sq2,   -2.0f,  sq2/2, 1.0f, 0.0f},
    //     {0.0f,  sq2,   -2.0f, -sq2/2, 1.0f, 0.0f},
    //     {0.0f, -sq2/2, -0.5f,  sq2,   1.0f, 0.0f},
    //     {0.0f,  sq2/2, -0.5f, -sq2,   1.0f, 0.0f},
    //     {0.0f,  1.0f,   0.0f,  -2.5f, 0.0f, 1.0f}
    // };

    // 0 =  r00 + r04 - 2.5f * r02
    // 1 = -(sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 2 =  (sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 3 =  (sq2 * r03 - sq2_d2 * r01) + (r04 - 0.5f * r02)
    // 4 = -(sq2 * r03 - sq2_d2 * r01) + (r04 - 0.5f * r02)
    // 5 =  r01 + r05 - 2.5f * r03

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 1) / 4;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __SSE2__
#if __AVX__
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
        float tmp[6][6][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 4) + (tj * 4) * elempack;

            __m512 _vm2_5 = _mm512_set1_ps(-2.5f);
            __m512 _vsq2 = _mm512_set1_ps(sq2);
            __m512 _vmsq2_d2 = _mm512_set1_ps(-sq2_d2);
            __m512 _vm2 = _mm512_set1_ps(-2.f);
            __m512 _vm0_5 = _mm512_set1_ps(-0.5f);

            for (int m = 0; m < 6; m++)
            {
                __m512 _r0 = _mm512_setzero_ps();
                __m512 _r1 = _mm512_setzero_ps();
                __m512 _r2 = _mm512_setzero_ps();
                __m512 _r3 = _mm512_setzero_ps();
                __m512 _r4 = _mm512_setzero_ps();
                __m512 _r5 = _mm512_setzero_ps();

                if (ti * 4 + m < h)
                {
                    if (elempack == 16)
                    {
                        _r0 = _mm512_load_ps(r0);
                        if (tj * 4 + 1 < w) _r1 = _mm512_load_ps(r0 + 16);
                        if (tj * 4 + 2 < w) _r2 = _mm512_load_ps(r0 + 32);
                        if (tj * 4 + 3 < w) _r3 = _mm512_load_ps(r0 + 48);
                        if (tj * 4 + 4 < w) _r4 = _mm512_load_ps(r0 + 64);
                        if (tj * 4 + 5 < w) _r5 = _mm512_load_ps(r0 + 80);
                    }
                    if (elempack == 8)
                    {
                        const float* r1 = r0 + N;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0)), _mm256_load_ps(r1), 1);
                        if (tj * 4 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 8)), _mm256_load_ps(r1 + 8), 1);
                        if (tj * 4 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 16)), _mm256_load_ps(r1 + 16), 1);
                        if (tj * 4 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 24)), _mm256_load_ps(r1 + 24), 1);
                        if (tj * 4 + 4 < w) _r4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 32)), _mm256_load_ps(r1 + 32), 1);
                        if (tj * 4 + 5 < w) _r5 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 40)), _mm256_load_ps(r1 + 40), 1);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2)), _mm_load_ps(r3), 1), 1);
                        if (tj * 4 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 4)), _mm_load_ps(r3 + 4), 1), 1);
                        if (tj * 4 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 8)), _mm_load_ps(r3 + 8), 1), 1);
                        if (tj * 4 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 12)), _mm_load_ps(r3 + 12), 1), 1);
                        if (tj * 4 + 4 < w) _r4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 16)), _mm_load_ps(r1 + 16), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 16)), _mm_load_ps(r3 + 16), 1), 1);
                        if (tj * 4 + 5 < w) _r5 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 20)), _mm_load_ps(r1 + 20), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 20)), _mm_load_ps(r3 + 20), 1), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;
                        const float* r8 = r0 + N * 8;
                        const float* r9 = r0 + N * 9;
                        const float* ra = r0 + N * 10;
                        const float* rb = r0 + N * 11;
                        const float* rc = r0 + N * 12;
                        const float* rd = r0 + N * 13;
                        const float* re = r0 + N * 14;
                        const float* rf = r0 + N * 15;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);
                        __m128 _t4 = _mm_loadu_ps(r4);
                        __m128 _t5 = _mm_loadu_ps(r5);
                        __m128 _t6 = _mm_loadu_ps(r6);
                        __m128 _t7 = _mm_loadu_ps(r7);
                        __m128 _t8 = _mm_loadu_ps(r8);
                        __m128 _t9 = _mm_loadu_ps(r9);
                        __m128 _ta = _mm_loadu_ps(ra);
                        __m128 _tb = _mm_loadu_ps(rb);
                        __m128 _tc = _mm_loadu_ps(rc);
                        __m128 _td = _mm_loadu_ps(rd);
                        __m128 _te = _mm_loadu_ps(re);
                        __m128 _tf = _mm_loadu_ps(rf);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);
                        _MM_TRANSPOSE4_PS(_t4, _t5, _t6, _t7);
                        _MM_TRANSPOSE4_PS(_t8, _t9, _ta, _tb);
                        _MM_TRANSPOSE4_PS(_tc, _td, _te, _tf);

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t0), _t4, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_t8), _tc, 1), 1);
                        if (tj * 4 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t1), _t5, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_t9), _td, 1), 1);
                        if (tj * 4 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t2), _t6, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_ta), _te, 1), 1);
                        if (tj * 4 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t3), _t7, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_tb), _tf, 1), 1);
                        if (tj * 4 + 4 < w) _r4 = _mm512_set_ps(rf[4], re[4], rd[4], rc[4], rb[4], ra[4], r9[4], r8[4], r7[4], r6[4], r5[4], r4[4], r3[4], r2[4], r1[4], r0[4]);
                        if (tj * 4 + 5 < w) _r5 = _mm512_set_ps(rf[5], re[5], rd[5], rc[5], rb[5], ra[5], r9[5], r8[5], r7[5], r6[5], r5[5], r4[5], r3[5], r2[5], r1[5], r0[5]);
                    }
                }

                __m512 _tmp12a = _mm512_fmadd_ps(_vmsq2_d2, _r3, _mm512_mul_ps(_r1, _vsq2));
                __m512 _tmp12b = _mm512_fmadd_ps(_vm2, _r2, _r4);
                __m512 _tmp34a = _mm512_fmadd_ps(_vmsq2_d2, _r1, _mm512_mul_ps(_r3, _vsq2));
                __m512 _tmp34b = _mm512_fmadd_ps(_vm0_5, _r2, _r4);

                __m512 _tmp0 = _mm512_fmadd_ps(_vm2_5, _r2, _mm512_add_ps(_r0, _r4));
                __m512 _tmp1 = _mm512_sub_ps(_tmp12b, _tmp12a);
                __m512 _tmp2 = _mm512_add_ps(_tmp12b, _tmp12a);
                __m512 _tmp3 = _mm512_add_ps(_tmp34b, _tmp34a);
                __m512 _tmp4 = _mm512_sub_ps(_tmp34b, _tmp34a);
                __m512 _tmp5 = _mm512_fmadd_ps(_vm2_5, _r3, _mm512_add_ps(_r1, _r5));

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);
                _mm512_store_ps(tmp[2][m], _tmp2);
                _mm512_store_ps(tmp[3][m], _tmp3);
                _mm512_store_ps(tmp[4][m], _tmp4);
                _mm512_store_ps(tmp[5][m], _tmp5);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 16;
            float* p1 = p0 + max_jj * 16;
            float* p2 = p0 + max_jj * 16 * 2;
            float* p3 = p0 + max_jj * 16 * 3;
            float* p4 = p0 + max_jj * 16 * 4;
            float* p5 = p0 + max_jj * 16 * 5;

            for (int m = 0; m < 6; m++)
            {
                __m512 _r0 = _mm512_load_ps(tmp[m][0]);
                __m512 _r1 = _mm512_load_ps(tmp[m][1]);
                __m512 _r2 = _mm512_load_ps(tmp[m][2]);
                __m512 _r3 = _mm512_load_ps(tmp[m][3]);
                __m512 _r4 = _mm512_load_ps(tmp[m][4]);
                __m512 _r5 = _mm512_load_ps(tmp[m][5]);

                __m512 _tmp12a = _mm512_fmadd_ps(_vmsq2_d2, _r3, _mm512_mul_ps(_r1, _vsq2));
                __m512 _tmp12b = _mm512_fmadd_ps(_vm2, _r2, _r4);
                __m512 _tmp34a = _mm512_fmadd_ps(_vmsq2_d2, _r1, _mm512_mul_ps(_r3, _vsq2));
                __m512 _tmp34b = _mm512_fmadd_ps(_vm0_5, _r2, _r4);

                __m512 _tmp0 = _mm512_fmadd_ps(_vm2_5, _r2, _mm512_add_ps(_r0, _r4));
                __m512 _tmp1 = _mm512_sub_ps(_tmp12b, _tmp12a);
                __m512 _tmp2 = _mm512_add_ps(_tmp12b, _tmp12a);
                __m512 _tmp3 = _mm512_add_ps(_tmp34b, _tmp34a);
                __m512 _tmp4 = _mm512_sub_ps(_tmp34b, _tmp34a);
                __m512 _tmp5 = _mm512_fmadd_ps(_vm2_5, _r3, _mm512_add_ps(_r1, _r5));

                _mm512_store_ps(p0, _tmp0);
                _mm512_store_ps(p1, _tmp1);
                _mm512_store_ps(p2, _tmp2);
                _mm512_store_ps(p3, _tmp3);
                _mm512_store_ps(p4, _tmp4);
                _mm512_store_ps(p5, _tmp5);

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
        float tmp[6][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 4) + (tj * 4) * elempack;

            __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
            __m256 _vsq2 = _mm256_set1_ps(sq2);
            __m256 _vmsq2_d2 = _mm256_set1_ps(-sq2_d2);
            __m256 _vm2 = _mm256_set1_ps(-2.f);
            __m256 _vm0_5 = _mm256_set1_ps(-0.5f);

            for (int m = 0; m < 6; m++)
            {
                __m256 _r0 = _mm256_setzero_ps();
                __m256 _r1 = _mm256_setzero_ps();
                __m256 _r2 = _mm256_setzero_ps();
                __m256 _r3 = _mm256_setzero_ps();
                __m256 _r4 = _mm256_setzero_ps();
                __m256 _r5 = _mm256_setzero_ps();

                if (ti * 4 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = _mm256_load_ps(r0);
                        if (tj * 4 + 1 < w) _r1 = _mm256_load_ps(r0 + 8);
                        if (tj * 4 + 2 < w) _r2 = _mm256_load_ps(r0 + 16);
                        if (tj * 4 + 3 < w) _r3 = _mm256_load_ps(r0 + 24);
                        if (tj * 4 + 4 < w) _r4 = _mm256_load_ps(r0 + 32);
                        if (tj * 4 + 5 < w) _r5 = _mm256_load_ps(r0 + 40);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1);
                        if (tj * 4 + 1 < w) _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1);
                        if (tj * 4 + 2 < w) _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1);
                        if (tj * 4 + 3 < w) _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1);
                        if (tj * 4 + 4 < w) _r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 16)), _mm_load_ps(r1 + 16), 1);
                        if (tj * 4 + 5 < w) _r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 20)), _mm_load_ps(r1 + 20), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);
                        __m128 _t4 = _mm_loadu_ps(r4);
                        __m128 _t5 = _mm_loadu_ps(r5);
                        __m128 _t6 = _mm_loadu_ps(r6);
                        __m128 _t7 = _mm_loadu_ps(r7);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);
                        _MM_TRANSPOSE4_PS(_t4, _t5, _t6, _t7);

                        _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t0), _t4, 1);
                        if (tj * 4 + 1 < w) _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t1), _t5, 1);
                        if (tj * 4 + 2 < w) _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t2), _t6, 1);
                        if (tj * 4 + 3 < w) _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t3), _t7, 1);
                        if (tj * 4 + 4 < w) _r4 = _mm256_set_ps(r7[4], r6[4], r5[4], r4[4], r3[4], r2[4], r1[4], r0[4]);
                        if (tj * 4 + 5 < w) _r5 = _mm256_set_ps(r7[5], r6[5], r5[5], r4[5], r3[5], r2[5], r1[5], r0[5]);
                    }
                }

                __m256 _tmp12a = _mm256_comp_fmadd_ps(_vmsq2_d2, _r3, _mm256_mul_ps(_r1, _vsq2));
                __m256 _tmp12b = _mm256_comp_fmadd_ps(_vm2, _r2, _r4);
                __m256 _tmp34a = _mm256_comp_fmadd_ps(_vmsq2_d2, _r1, _mm256_mul_ps(_r3, _vsq2));
                __m256 _tmp34b = _mm256_comp_fmadd_ps(_vm0_5, _r2, _r4);

                __m256 _tmp0 = _mm256_comp_fmadd_ps(_vm2_5, _r2, _mm256_add_ps(_r0, _r4));
                __m256 _tmp1 = _mm256_sub_ps(_tmp12b, _tmp12a);
                __m256 _tmp2 = _mm256_add_ps(_tmp12b, _tmp12a);
                __m256 _tmp3 = _mm256_add_ps(_tmp34b, _tmp34a);
                __m256 _tmp4 = _mm256_sub_ps(_tmp34b, _tmp34a);
                __m256 _tmp5 = _mm256_comp_fmadd_ps(_vm2_5, _r3, _mm256_add_ps(_r1, _r5));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm256_storeu_ps(tmp[0][m], _tmp0);
                _mm256_storeu_ps(tmp[1][m], _tmp1);
                _mm256_storeu_ps(tmp[2][m], _tmp2);
                _mm256_storeu_ps(tmp[3][m], _tmp3);
                _mm256_storeu_ps(tmp[4][m], _tmp4);
                _mm256_storeu_ps(tmp[5][m], _tmp5);
#else
                _mm256_store_ps(tmp[0][m], _tmp0);
                _mm256_store_ps(tmp[1][m], _tmp1);
                _mm256_store_ps(tmp[2][m], _tmp2);
                _mm256_store_ps(tmp[3][m], _tmp3);
                _mm256_store_ps(tmp[4][m], _tmp4);
                _mm256_store_ps(tmp[5][m], _tmp5);
#endif

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;
            float* p4 = p0 + max_jj * 8 * 4;
            float* p5 = p0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m256 _r0 = _mm256_loadu_ps(tmp[m][0]);
                __m256 _r1 = _mm256_loadu_ps(tmp[m][1]);
                __m256 _r2 = _mm256_loadu_ps(tmp[m][2]);
                __m256 _r3 = _mm256_loadu_ps(tmp[m][3]);
                __m256 _r4 = _mm256_loadu_ps(tmp[m][4]);
                __m256 _r5 = _mm256_loadu_ps(tmp[m][5]);
#else
                __m256 _r0 = _mm256_load_ps(tmp[m][0]);
                __m256 _r1 = _mm256_load_ps(tmp[m][1]);
                __m256 _r2 = _mm256_load_ps(tmp[m][2]);
                __m256 _r3 = _mm256_load_ps(tmp[m][3]);
                __m256 _r4 = _mm256_load_ps(tmp[m][4]);
                __m256 _r5 = _mm256_load_ps(tmp[m][5]);
#endif

                __m256 _tmp12a = _mm256_comp_fmadd_ps(_vmsq2_d2, _r3, _mm256_mul_ps(_r1, _vsq2));
                __m256 _tmp12b = _mm256_comp_fmadd_ps(_vm2, _r2, _r4);
                __m256 _tmp34a = _mm256_comp_fmadd_ps(_vmsq2_d2, _r1, _mm256_mul_ps(_r3, _vsq2));
                __m256 _tmp34b = _mm256_comp_fmadd_ps(_vm0_5, _r2, _r4);

                __m256 _tmp0 = _mm256_comp_fmadd_ps(_vm2_5, _r2, _mm256_add_ps(_r0, _r4));
                __m256 _tmp1 = _mm256_sub_ps(_tmp12b, _tmp12a);
                __m256 _tmp2 = _mm256_add_ps(_tmp12b, _tmp12a);
                __m256 _tmp3 = _mm256_add_ps(_tmp34b, _tmp34a);
                __m256 _tmp4 = _mm256_sub_ps(_tmp34b, _tmp34a);
                __m256 _tmp5 = _mm256_comp_fmadd_ps(_vm2_5, _r3, _mm256_add_ps(_r1, _r5));

                _mm256_store_ps(p0, _tmp0);
                _mm256_store_ps(p1, _tmp1);
                _mm256_store_ps(p2, _tmp2);
                _mm256_store_ps(p3, _tmp3);
                _mm256_store_ps(p4, _tmp4);
                _mm256_store_ps(p5, _tmp5);

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
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
#else // __AVX__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __AVX__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 4) + (tj * 4) * elempack;

            __m128 _vm2_5 = _mm_set1_ps(-2.5f);
            __m128 _vsq2 = _mm_set1_ps(sq2);
            __m128 _vmsq2_d2 = _mm_set1_ps(-sq2_d2);
            __m128 _vm2 = _mm_set1_ps(-2.f);
            __m128 _vm0_5 = _mm_set1_ps(-0.5f);

            for (int m = 0; m < 6; m++)
            {
                __m128 _r0 = _mm_setzero_ps();
                __m128 _r1 = _mm_setzero_ps();
                __m128 _r2 = _mm_setzero_ps();
                __m128 _r3 = _mm_setzero_ps();
                __m128 _r4 = _mm_setzero_ps();
                __m128 _r5 = _mm_setzero_ps();

                if (ti * 4 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = _mm_load_ps(r0);
                        if (tj * 4 + 1 < w) _r1 = _mm_load_ps(r0 + 4);
                        if (tj * 4 + 2 < w) _r2 = _mm_load_ps(r0 + 8);
                        if (tj * 4 + 3 < w) _r3 = _mm_load_ps(r0 + 12);
                        if (tj * 4 + 4 < w) _r4 = _mm_load_ps(r0 + 16);
                        if (tj * 4 + 5 < w) _r5 = _mm_load_ps(r0 + 20);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 4 + 1 < w) _r1 = _t1;
                        if (tj * 4 + 2 < w) _r2 = _t2;
                        if (tj * 4 + 3 < w) _r3 = _t3;
                        if (tj * 4 + 4 < w) _r4 = _mm_set_ps(r3[4], r2[4], r1[4], r0[4]);
                        if (tj * 4 + 5 < w) _r5 = _mm_set_ps(r3[5], r2[5], r1[5], r0[5]);
                    }
                }

                __m128 _tmp12a = _mm_comp_fmadd_ps(_vmsq2_d2, _r3, _mm_mul_ps(_r1, _vsq2));
                __m128 _tmp12b = _mm_comp_fmadd_ps(_vm2, _r2, _r4);
                __m128 _tmp34a = _mm_comp_fmadd_ps(_vmsq2_d2, _r1, _mm_mul_ps(_r3, _vsq2));
                __m128 _tmp34b = _mm_comp_fmadd_ps(_vm0_5, _r2, _r4);

                __m128 _tmp0 = _mm_comp_fmadd_ps(_vm2_5, _r2, _mm_add_ps(_r0, _r4));
                __m128 _tmp1 = _mm_sub_ps(_tmp12b, _tmp12a);
                __m128 _tmp2 = _mm_add_ps(_tmp12b, _tmp12a);
                __m128 _tmp3 = _mm_add_ps(_tmp34b, _tmp34a);
                __m128 _tmp4 = _mm_sub_ps(_tmp34b, _tmp34a);
                __m128 _tmp5 = _mm_comp_fmadd_ps(_vm2_5, _r3, _mm_add_ps(_r1, _r5));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_ps(tmp[0][m], _tmp0);
                _mm_storeu_ps(tmp[1][m], _tmp1);
                _mm_storeu_ps(tmp[2][m], _tmp2);
                _mm_storeu_ps(tmp[3][m], _tmp3);
                _mm_storeu_ps(tmp[4][m], _tmp4);
                _mm_storeu_ps(tmp[5][m], _tmp5);
#else
                _mm_store_ps(tmp[0][m], _tmp0);
                _mm_store_ps(tmp[1][m], _tmp1);
                _mm_store_ps(tmp[2][m], _tmp2);
                _mm_store_ps(tmp[3][m], _tmp3);
                _mm_store_ps(tmp[4][m], _tmp4);
                _mm_store_ps(tmp[5][m], _tmp5);
#endif

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;
            float* p4 = p0 + max_jj * 4 * 4;
            float* p5 = p0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128 _r0 = _mm_loadu_ps(tmp[m][0]);
                __m128 _r1 = _mm_loadu_ps(tmp[m][1]);
                __m128 _r2 = _mm_loadu_ps(tmp[m][2]);
                __m128 _r3 = _mm_loadu_ps(tmp[m][3]);
                __m128 _r4 = _mm_loadu_ps(tmp[m][4]);
                __m128 _r5 = _mm_loadu_ps(tmp[m][5]);
#else
                __m128 _r0 = _mm_load_ps(tmp[m][0]);
                __m128 _r1 = _mm_load_ps(tmp[m][1]);
                __m128 _r2 = _mm_load_ps(tmp[m][2]);
                __m128 _r3 = _mm_load_ps(tmp[m][3]);
                __m128 _r4 = _mm_load_ps(tmp[m][4]);
                __m128 _r5 = _mm_load_ps(tmp[m][5]);
#endif

                __m128 _tmp12a = _mm_comp_fmadd_ps(_vmsq2_d2, _r3, _mm_mul_ps(_r1, _vsq2));
                __m128 _tmp12b = _mm_comp_fmadd_ps(_vm2, _r2, _r4);
                __m128 _tmp34a = _mm_comp_fmadd_ps(_vmsq2_d2, _r1, _mm_mul_ps(_r3, _vsq2));
                __m128 _tmp34b = _mm_comp_fmadd_ps(_vm0_5, _r2, _r4);

                __m128 _tmp0 = _mm_comp_fmadd_ps(_vm2_5, _r2, _mm_add_ps(_r0, _r4));
                __m128 _tmp1 = _mm_sub_ps(_tmp12b, _tmp12a);
                __m128 _tmp2 = _mm_add_ps(_tmp12b, _tmp12a);
                __m128 _tmp3 = _mm_add_ps(_tmp34b, _tmp34a);
                __m128 _tmp4 = _mm_sub_ps(_tmp34b, _tmp34a);
                __m128 _tmp5 = _mm_comp_fmadd_ps(_vm2_5, _r3, _mm_add_ps(_r1, _r5));

                _mm_store_ps(p0, _tmp0);
                _mm_store_ps(p1, _tmp1);
                _mm_store_ps(p2, _tmp2);
                _mm_store_ps(p3, _tmp3);
                _mm_store_ps(p4, _tmp4);
                _mm_store_ps(p5, _tmp5);

                p0 += max_jj * 6 * 4;
                p1 += max_jj * 6 * 4;
                p2 += max_jj * 6 * 4;
                p3 += max_jj * 6 * 4;
                p4 += max_jj * 6 * 4;
                p5 += max_jj * 6 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __SSE2__
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __SSE2__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        float tmp[6][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
                float r40 = 0.f;
                float r41 = 0.f;
                float r50 = 0.f;
                float r51 = 0.f;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

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

                float tmp12a0 = sq2 * r10 - sq2_d2 * r30;
                float tmp12a1 = sq2 * r11 - sq2_d2 * r31;
                float tmp12b0 = r40 - 2 * r20;
                float tmp12b1 = r41 - 2 * r21;
                float tmp34a0 = sq2 * r30 - sq2_d2 * r10;
                float tmp34a1 = sq2 * r31 - sq2_d2 * r11;
                float tmp34b0 = r40 - 0.5f * r20;
                float tmp34b1 = r41 - 0.5f * r21;

                tmp[0][m][0] = r00 + r40 - 2.5f * r20;
                tmp[0][m][1] = r01 + r41 - 2.5f * r21;
                tmp[1][m][0] = tmp12b0 - tmp12a0;
                tmp[1][m][1] = tmp12b1 - tmp12a1;
                tmp[2][m][0] = tmp12b0 + tmp12a0;
                tmp[2][m][1] = tmp12b1 + tmp12a1;
                tmp[3][m][0] = tmp34b0 + tmp34a0;
                tmp[3][m][1] = tmp34b1 + tmp34a1;
                tmp[4][m][0] = tmp34b0 - tmp34a0;
                tmp[4][m][1] = tmp34b1 - tmp34a1;
                tmp[5][m][0] = r10 + r50 - 2.5f * r30;
                tmp[5][m][1] = r11 + r51 - 2.5f * r31;

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;
            float* p4 = p0 + max_jj * 2 * 4;
            float* p5 = p0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];

                float tmp12a0 = sq2 * r10 - sq2_d2 * r30;
                float tmp12a1 = sq2 * r11 - sq2_d2 * r31;
                float tmp12b0 = r40 - 2 * r20;
                float tmp12b1 = r41 - 2 * r21;
                float tmp34a0 = sq2 * r30 - sq2_d2 * r10;
                float tmp34a1 = sq2 * r31 - sq2_d2 * r11;
                float tmp34b0 = r40 - 0.5f * r20;
                float tmp34b1 = r41 - 0.5f * r21;

                p0[0] = r00 + r40 - 2.5f * r20;
                p0[1] = r01 + r41 - 2.5f * r21;
                p1[0] = tmp12b0 - tmp12a0;
                p1[1] = tmp12b1 - tmp12a1;
                p2[0] = tmp12b0 + tmp12a0;
                p2[1] = tmp12b1 + tmp12a1;
                p3[0] = tmp34b0 + tmp34a0;
                p3[1] = tmp34b1 + tmp34a1;
                p4[0] = tmp34b0 - tmp34a0;
                p4[1] = tmp34b1 - tmp34a1;
                p5[0] = r10 + r50 - 2.5f * r30;
                p5[1] = r11 + r51 - 2.5f * r31;

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
        float tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;
                float r4 = 0.f;
                float r5 = 0.f;

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

                float tmp12a = sq2 * r1 - sq2_d2 * r3;
                float tmp12b = r4 - 2 * r2;
                float tmp34a = sq2 * r3 - sq2_d2 * r1;
                float tmp34b = r4 - 0.5f * r2;

                tmp[0][m] = r0 + r4 - 2.5f * r2;
                tmp[1][m] = tmp12b - tmp12a;
                tmp[2][m] = tmp12b + tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r1 + r5 - 2.5f * r3;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;
            float* p4 = p0 + max_jj * 4;
            float* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float tmp12a = sq2 * r1 - sq2_d2 * r3;
                float tmp12b = r4 - 2 * r2;
                float tmp34a = sq2 * r3 - sq2_d2 * r1;
                float tmp34b = r4 - 0.5f * r2;

                p0[0] = r0 + r4 - 2.5f * r2;
                p1[0] = tmp12b - tmp12a;
                p2[0] = tmp12b + tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r1 + r5 - 2.5f * r3;

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

static inline void conv3x3s1_winograd43_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    const float sq2 = 1.41421356237;
    const float sq2_m2 = 1.41421356237 * 2;
    const float sq2_d2 = 1.41421356237 / 2;
    const float sq2_d4 = 1.41421356237 / 4;

    // const float otm[4][6] = {
    //     {1.0f, 1.0f,   1.0f,  1.0f,  1.0f,   0.0f},
    //     {0.0f, sq2/2, -sq2/2, sq2,   -sq2,   0.0f},
    //     {0.0f, 0.5f,   0.5f,  2.0f,  2.0f,   0.0f},
    //     {0.0f, sq2/4, -sq2/4, sq2*2, -sq2*2, 1.0f}
    // };

    // 0 = r00 + (r01 + r02) + (r03 + r04)
    // 1 =       (r01 - r02) * sq2_d2 + (r03 - r04) * sq2
    // 2 =       (r01 + r02) * 0.5f + (r03 + r04) * 2
    // 3 = r05 + (r01 - r02) * sq2_d4 + (r03 - r04) * sq2_m2

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 3) / 4;

    const float* biasptr = bias;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        __m512 _bias0 = biasptr ? _mm512_loadu_ps(biasptr + i + ii) : _mm512_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[4][6][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 16;
            const float* r1 = r0 + max_jj * 16;
            const float* r2 = r0 + max_jj * 16 * 2;
            const float* r3 = r0 + max_jj * 16 * 3;
            const float* r4 = r0 + max_jj * 16 * 4;
            const float* r5 = r0 + max_jj * 16 * 5;

            __m512 _vsq2 = _mm512_set1_ps(sq2);
            __m512 _vsq2_d2 = _mm512_set1_ps(sq2_d2);
            __m512 _vsq2_d4 = _mm512_set1_ps(sq2_d4);
            __m512 _vsq2_m2 = _mm512_set1_ps(sq2_m2);
            __m512 _v0_5 = _mm512_set1_ps(0.5f);
            __m512 _v2 = _mm512_set1_ps(2.f);

            for (int m = 0; m < 6; m++)
            {
                __m512 _r0 = _mm512_load_ps(r0);
                __m512 _r1 = _mm512_load_ps(r1);
                __m512 _r2 = _mm512_load_ps(r2);
                __m512 _r3 = _mm512_load_ps(r3);
                __m512 _r4 = _mm512_load_ps(r4);
                __m512 _r5 = _mm512_load_ps(r5);

                __m512 _tmp02a = _mm512_add_ps(_r1, _r2);
                __m512 _tmp02b = _mm512_add_ps(_r3, _r4);
                __m512 _tmp13a = _mm512_sub_ps(_r1, _r2);
                __m512 _tmp13b = _mm512_sub_ps(_r3, _r4);

                __m512 _tmp0 = _mm512_add_ps(_mm512_add_ps(_r0, _tmp02a), _tmp02b);
                __m512 _tmp1 = _mm512_fmadd_ps(_tmp13b, _vsq2, _mm512_mul_ps(_tmp13a, _vsq2_d2));
                __m512 _tmp2 = _mm512_fmadd_ps(_tmp02b, _v2, _mm512_mul_ps(_tmp02a, _v0_5));
                __m512 _tmp3 = _mm512_fmadd_ps(_tmp13b, _vsq2_m2, _mm512_fmadd_ps(_tmp13a, _vsq2_d4, _r5));

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);
                _mm512_store_ps(tmp[2][m], _tmp2);
                _mm512_store_ps(tmp[3][m], _tmp3);

                r0 += max_jj * 6 * 16;
                r1 += max_jj * 6 * 16;
                r2 += max_jj * 6 * 16;
                r3 += max_jj * 6 * 16;
                r4 += max_jj * 6 * 16;
                r5 += max_jj * 6 * 16;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __m512 _r0 = _mm512_load_ps(tmp[m][0]);
                __m512 _r1 = _mm512_load_ps(tmp[m][1]);
                __m512 _r2 = _mm512_load_ps(tmp[m][2]);
                __m512 _r3 = _mm512_load_ps(tmp[m][3]);
                __m512 _r4 = _mm512_load_ps(tmp[m][4]);
                __m512 _r5 = _mm512_load_ps(tmp[m][5]);

                __m512 _tmp02a = _mm512_add_ps(_r1, _r2);
                __m512 _tmp02b = _mm512_add_ps(_r3, _r4);
                __m512 _tmp13a = _mm512_sub_ps(_r1, _r2);
                __m512 _tmp13b = _mm512_sub_ps(_r3, _r4);

                __m512 _tmp0 = _mm512_add_ps(_mm512_add_ps(_r0, _tmp02a), _mm512_add_ps(_tmp02b, _bias0));
                __m512 _tmp1 = _mm512_fmadd_ps(_tmp13b, _vsq2, _mm512_fmadd_ps(_tmp13a, _vsq2_d2, _bias0));
                __m512 _tmp2 = _mm512_fmadd_ps(_tmp02b, _v2, _mm512_fmadd_ps(_tmp02a, _v0_5, _bias0));
                __m512 _tmp3 = _mm512_fmadd_ps(_tmp13b, _vsq2_m2, _mm512_fmadd_ps(_tmp13a, _vsq2_d4, _mm512_add_ps(_r5, _bias0)));

                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm512_store_ps(outptr0 + 16, _tmp1);
                    if (tj * 4 + 2 < outw) _mm512_store_ps(outptr0 + 32, _tmp2);
                    if (tj * 4 + 3 < outw) _mm512_store_ps(outptr0 + 48, _tmp3);
                }
                if (out_elempack == 8)
                {
                    float* outptr1 = outptr0 + N;

                    _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_tmp0, 0));
                    _mm256_store_ps(outptr1, _mm512_extractf32x8_ps(_tmp0, 1));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm256_store_ps(outptr0 + 8, _mm512_extractf32x8_ps(_tmp1, 0));
                        _mm256_store_ps(outptr1 + 8, _mm512_extractf32x8_ps(_tmp1, 1));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm256_store_ps(outptr0 + 16, _mm512_extractf32x8_ps(_tmp2, 0));
                        _mm256_store_ps(outptr1 + 16, _mm512_extractf32x8_ps(_tmp2, 1));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm256_store_ps(outptr0 + 24, _mm512_extractf32x8_ps(_tmp3, 0));
                        _mm256_store_ps(outptr1 + 24, _mm512_extractf32x8_ps(_tmp3, 1));
                    }
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_tmp0, 0));
                    _mm_store_ps(outptr1, _mm512_extractf32x4_ps(_tmp0, 1));
                    _mm_store_ps(outptr2, _mm512_extractf32x4_ps(_tmp0, 2));
                    _mm_store_ps(outptr3, _mm512_extractf32x4_ps(_tmp0, 3));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm_store_ps(outptr0 + 4, _mm512_extractf32x4_ps(_tmp1, 0));
                        _mm_store_ps(outptr1 + 4, _mm512_extractf32x4_ps(_tmp1, 1));
                        _mm_store_ps(outptr2 + 4, _mm512_extractf32x4_ps(_tmp1, 2));
                        _mm_store_ps(outptr3 + 4, _mm512_extractf32x4_ps(_tmp1, 3));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm_store_ps(outptr0 + 8, _mm512_extractf32x4_ps(_tmp2, 0));
                        _mm_store_ps(outptr1 + 8, _mm512_extractf32x4_ps(_tmp2, 1));
                        _mm_store_ps(outptr2 + 8, _mm512_extractf32x4_ps(_tmp2, 2));
                        _mm_store_ps(outptr3 + 8, _mm512_extractf32x4_ps(_tmp2, 3));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm_store_ps(outptr0 + 12, _mm512_extractf32x4_ps(_tmp3, 0));
                        _mm_store_ps(outptr1 + 12, _mm512_extractf32x4_ps(_tmp3, 1));
                        _mm_store_ps(outptr2 + 12, _mm512_extractf32x4_ps(_tmp3, 2));
                        _mm_store_ps(outptr3 + 12, _mm512_extractf32x4_ps(_tmp3, 3));
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[16];
                    float tmp1[16];
                    float tmp2[16];
                    float tmp3[16];
                    _mm512_storeu_ps(tmp0, _tmp0);
                    _mm512_storeu_ps(tmp1, _tmp1);
                    _mm512_storeu_ps(tmp2, _tmp2);
                    _mm512_storeu_ps(tmp3, _tmp3);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;
                    float* outptr8 = outptr0 + N * 8;
                    float* outptr9 = outptr0 + N * 9;
                    float* outptra = outptr0 + N * 10;
                    float* outptrb = outptr0 + N * 11;
                    float* outptrc = outptr0 + N * 12;
                    float* outptrd = outptr0 + N * 13;
                    float* outptre = outptr0 + N * 14;
                    float* outptrf = outptr0 + N * 15;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    outptr8[0] = tmp0[8];
                    outptr9[0] = tmp0[9];
                    outptra[0] = tmp0[10];
                    outptrb[0] = tmp0[11];
                    outptrc[0] = tmp0[12];
                    outptrd[0] = tmp0[13];
                    outptre[0] = tmp0[14];
                    outptrf[0] = tmp0[15];
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
                        outptr8[1] = tmp1[8];
                        outptr9[1] = tmp1[9];
                        outptra[1] = tmp1[10];
                        outptrb[1] = tmp1[11];
                        outptrc[1] = tmp1[12];
                        outptrd[1] = tmp1[13];
                        outptre[1] = tmp1[14];
                        outptrf[1] = tmp1[15];
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
                        outptr8[2] = tmp2[8];
                        outptr9[2] = tmp2[9];
                        outptra[2] = tmp2[10];
                        outptrb[2] = tmp2[11];
                        outptrc[2] = tmp2[12];
                        outptrd[2] = tmp2[13];
                        outptre[2] = tmp2[14];
                        outptrf[2] = tmp2[15];
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
                        outptr8[3] = tmp3[8];
                        outptr9[3] = tmp3[9];
                        outptra[3] = tmp3[10];
                        outptrb[3] = tmp3[11];
                        outptrc[3] = tmp3[12];
                        outptrd[3] = tmp3[13];
                        outptre[3] = tmp3[14];
                        outptrf[3] = tmp3[15];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _bias0 = biasptr ? _mm256_loadu_ps(biasptr + i + ii) : _mm256_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(32))
#else
        __attribute__((aligned(32)))
#endif
        float tmp[4][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;
            const float* r4 = r0 + max_jj * 8 * 4;
            const float* r5 = r0 + max_jj * 8 * 5;

            __m256 _vsq2 = _mm256_set1_ps(sq2);
            __m256 _vsq2_d2 = _mm256_set1_ps(sq2_d2);
            __m256 _vsq2_d4 = _mm256_set1_ps(sq2_d4);
            __m256 _vsq2_m2 = _mm256_set1_ps(sq2_m2);
            __m256 _v0_5 = _mm256_set1_ps(0.5f);
            __m256 _v2 = _mm256_set1_ps(2.f);

            for (int m = 0; m < 6; m++)
            {
                __m256 _r0 = _mm256_load_ps(r0);
                __m256 _r1 = _mm256_load_ps(r1);
                __m256 _r2 = _mm256_load_ps(r2);
                __m256 _r3 = _mm256_load_ps(r3);
                __m256 _r4 = _mm256_load_ps(r4);
                __m256 _r5 = _mm256_load_ps(r5);

                __m256 _tmp02a = _mm256_add_ps(_r1, _r2);
                __m256 _tmp02b = _mm256_add_ps(_r3, _r4);
                __m256 _tmp13a = _mm256_sub_ps(_r1, _r2);
                __m256 _tmp13b = _mm256_sub_ps(_r3, _r4);

                __m256 _tmp0 = _mm256_add_ps(_mm256_add_ps(_r0, _tmp02a), _tmp02b);
                __m256 _tmp1 = _mm256_comp_fmadd_ps(_tmp13b, _vsq2, _mm256_mul_ps(_tmp13a, _vsq2_d2));
                __m256 _tmp2 = _mm256_comp_fmadd_ps(_tmp02b, _v2, _mm256_mul_ps(_tmp02a, _v0_5));
                __m256 _tmp3 = _mm256_comp_fmadd_ps(_tmp13b, _vsq2_m2, _mm256_comp_fmadd_ps(_tmp13a, _vsq2_d4, _r5));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm256_storeu_ps(tmp[0][m], _tmp0);
                _mm256_storeu_ps(tmp[1][m], _tmp1);
                _mm256_storeu_ps(tmp[2][m], _tmp2);
                _mm256_storeu_ps(tmp[3][m], _tmp3);
#else
                _mm256_store_ps(tmp[0][m], _tmp0);
                _mm256_store_ps(tmp[1][m], _tmp1);
                _mm256_store_ps(tmp[2][m], _tmp2);
                _mm256_store_ps(tmp[3][m], _tmp3);
#endif

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m256 _r0 = _mm256_loadu_ps(tmp[m][0]);
                __m256 _r1 = _mm256_loadu_ps(tmp[m][1]);
                __m256 _r2 = _mm256_loadu_ps(tmp[m][2]);
                __m256 _r3 = _mm256_loadu_ps(tmp[m][3]);
                __m256 _r4 = _mm256_loadu_ps(tmp[m][4]);
                __m256 _r5 = _mm256_loadu_ps(tmp[m][5]);
#else
                __m256 _r0 = _mm256_load_ps(tmp[m][0]);
                __m256 _r1 = _mm256_load_ps(tmp[m][1]);
                __m256 _r2 = _mm256_load_ps(tmp[m][2]);
                __m256 _r3 = _mm256_load_ps(tmp[m][3]);
                __m256 _r4 = _mm256_load_ps(tmp[m][4]);
                __m256 _r5 = _mm256_load_ps(tmp[m][5]);
#endif

                __m256 _tmp02a = _mm256_add_ps(_r1, _r2);
                __m256 _tmp02b = _mm256_add_ps(_r3, _r4);
                __m256 _tmp13a = _mm256_sub_ps(_r1, _r2);
                __m256 _tmp13b = _mm256_sub_ps(_r3, _r4);

                __m256 _tmp0 = _mm256_add_ps(_mm256_add_ps(_r0, _tmp02a), _mm256_add_ps(_tmp02b, _bias0));
                __m256 _tmp1 = _mm256_comp_fmadd_ps(_tmp13b, _vsq2, _mm256_comp_fmadd_ps(_tmp13a, _vsq2_d2, _bias0));
                __m256 _tmp2 = _mm256_comp_fmadd_ps(_tmp02b, _v2, _mm256_comp_fmadd_ps(_tmp02a, _v0_5, _bias0));
                __m256 _tmp3 = _mm256_comp_fmadd_ps(_tmp13b, _vsq2_m2, _mm256_comp_fmadd_ps(_tmp13a, _vsq2_d4, _mm256_add_ps(_r5, _bias0)));

                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm256_store_ps(outptr0 + 8, _tmp1);
                    if (tj * 4 + 2 < outw) _mm256_store_ps(outptr0 + 16, _tmp2);
                    if (tj * 4 + 3 < outw) _mm256_store_ps(outptr0 + 24, _tmp3);
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    _mm_store_ps(outptr0, _mm256_extractf128_ps(_tmp0, 0));
                    _mm_store_ps(outptr1, _mm256_extractf128_ps(_tmp0, 1));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm_store_ps(outptr0 + 4, _mm256_extractf128_ps(_tmp1, 0));
                        _mm_store_ps(outptr1 + 4, _mm256_extractf128_ps(_tmp1, 1));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm_store_ps(outptr0 + 8, _mm256_extractf128_ps(_tmp2, 0));
                        _mm_store_ps(outptr1 + 8, _mm256_extractf128_ps(_tmp2, 1));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm_store_ps(outptr0 + 12, _mm256_extractf128_ps(_tmp3, 0));
                        _mm_store_ps(outptr1 + 12, _mm256_extractf128_ps(_tmp3, 1));
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    float tmp2[8];
                    float tmp3[8];
                    _mm256_storeu_ps(tmp0, _tmp0);
                    _mm256_storeu_ps(tmp1, _tmp1);
                    _mm256_storeu_ps(tmp2, _tmp2);
                    _mm256_storeu_ps(tmp3, _tmp3);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

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
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _bias0 = biasptr ? _mm_loadu_ps(biasptr + i + ii) : _mm_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;
            const float* r4 = r0 + max_jj * 4 * 4;
            const float* r5 = r0 + max_jj * 4 * 5;

            __m128 _vsq2 = _mm_set1_ps(sq2);
            __m128 _vsq2_d2 = _mm_set1_ps(sq2_d2);
            __m128 _vsq2_d4 = _mm_set1_ps(sq2_d4);
            __m128 _vsq2_m2 = _mm_set1_ps(sq2_m2);
            __m128 _v0_5 = _mm_set1_ps(0.5f);
            __m128 _v2 = _mm_set1_ps(2.f);

            for (int m = 0; m < 6; m++)
            {
                __m128 _r0 = _mm_load_ps(r0);
                __m128 _r1 = _mm_load_ps(r1);
                __m128 _r2 = _mm_load_ps(r2);
                __m128 _r3 = _mm_load_ps(r3);
                __m128 _r4 = _mm_load_ps(r4);
                __m128 _r5 = _mm_load_ps(r5);

                __m128 _tmp02a = _mm_add_ps(_r1, _r2);
                __m128 _tmp02b = _mm_add_ps(_r3, _r4);
                __m128 _tmp13a = _mm_sub_ps(_r1, _r2);
                __m128 _tmp13b = _mm_sub_ps(_r3, _r4);

                __m128 _tmp0 = _mm_add_ps(_mm_add_ps(_r0, _tmp02a), _tmp02b);
                __m128 _tmp1 = _mm_comp_fmadd_ps(_tmp13b, _vsq2, _mm_mul_ps(_tmp13a, _vsq2_d2));
                __m128 _tmp2 = _mm_comp_fmadd_ps(_tmp02b, _v2, _mm_mul_ps(_tmp02a, _v0_5));
                __m128 _tmp3 = _mm_comp_fmadd_ps(_tmp13b, _vsq2_m2, _mm_comp_fmadd_ps(_tmp13a, _vsq2_d4, _r5));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_ps(tmp[0][m], _tmp0);
                _mm_storeu_ps(tmp[1][m], _tmp1);
                _mm_storeu_ps(tmp[2][m], _tmp2);
                _mm_storeu_ps(tmp[3][m], _tmp3);
#else
                _mm_store_ps(tmp[0][m], _tmp0);
                _mm_store_ps(tmp[1][m], _tmp1);
                _mm_store_ps(tmp[2][m], _tmp2);
                _mm_store_ps(tmp[3][m], _tmp3);
#endif

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128 _r0 = _mm_loadu_ps(tmp[m][0]);
                __m128 _r1 = _mm_loadu_ps(tmp[m][1]);
                __m128 _r2 = _mm_loadu_ps(tmp[m][2]);
                __m128 _r3 = _mm_loadu_ps(tmp[m][3]);
                __m128 _r4 = _mm_loadu_ps(tmp[m][4]);
                __m128 _r5 = _mm_loadu_ps(tmp[m][5]);
#else
                __m128 _r0 = _mm_load_ps(tmp[m][0]);
                __m128 _r1 = _mm_load_ps(tmp[m][1]);
                __m128 _r2 = _mm_load_ps(tmp[m][2]);
                __m128 _r3 = _mm_load_ps(tmp[m][3]);
                __m128 _r4 = _mm_load_ps(tmp[m][4]);
                __m128 _r5 = _mm_load_ps(tmp[m][5]);
#endif

                __m128 _tmp02a = _mm_add_ps(_r1, _r2);
                __m128 _tmp02b = _mm_add_ps(_r3, _r4);
                __m128 _tmp13a = _mm_sub_ps(_r1, _r2);
                __m128 _tmp13b = _mm_sub_ps(_r3, _r4);

                __m128 _tmp0 = _mm_add_ps(_mm_add_ps(_r0, _tmp02a), _mm_add_ps(_tmp02b, _bias0));
                __m128 _tmp1 = _mm_comp_fmadd_ps(_tmp13b, _vsq2, _mm_comp_fmadd_ps(_tmp13a, _vsq2_d2, _bias0));
                __m128 _tmp2 = _mm_comp_fmadd_ps(_tmp02b, _v2, _mm_comp_fmadd_ps(_tmp02a, _v0_5, _bias0));
                __m128 _tmp3 = _mm_comp_fmadd_ps(_tmp13b, _vsq2_m2, _mm_comp_fmadd_ps(_tmp13a, _vsq2_d4, _mm_add_ps(_r5, _bias0)));

                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm_store_ps(outptr0 + 4, _tmp1);
                    if (tj * 4 + 2 < outw) _mm_store_ps(outptr0 + 8, _tmp2);
                    if (tj * 4 + 3 < outw) _mm_store_ps(outptr0 + 12, _tmp3);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    float tmp2[4];
                    float tmp3[4];
                    _mm_storeu_ps(tmp0, _tmp0);
                    _mm_storeu_ps(tmp1, _tmp1);
                    _mm_storeu_ps(tmp2, _tmp2);
                    _mm_storeu_ps(tmp3, _tmp3);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

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
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

        float tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;
            const float* r4 = r0 + max_jj * 2 * 4;
            const float* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                float tmp02a0 = r1[0] + r2[0];
                float tmp02a1 = r1[1] + r2[1];
                float tmp02b0 = r3[0] + r4[0];
                float tmp02b1 = r3[1] + r4[1];
                float tmp13a0 = r1[0] - r2[0];
                float tmp13a1 = r1[1] - r2[1];
                float tmp13b0 = r3[0] - r4[0];
                float tmp13b1 = r3[1] - r4[1];

                tmp[0][m][0] = r0[0] + tmp02a0 + tmp02b0;
                tmp[0][m][1] = r0[1] + tmp02a1 + tmp02b1;
                tmp[1][m][0] = tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                tmp[1][m][1] = tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                tmp[2][m][0] = tmp02a0 * 0.5f + tmp02b0 * 2;
                tmp[2][m][1] = tmp02a1 * 0.5f + tmp02b1 * 2;
                tmp[3][m][0] = r5[0] + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                tmp[3][m][1] = r5[1] + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];

                float tmp02a0 = r10 + r20;
                float tmp02a1 = r11 + r21;
                float tmp02b0 = r30 + r40;
                float tmp02b1 = r31 + r41;
                float tmp13a0 = r10 - r20;
                float tmp13a1 = r11 - r21;
                float tmp13b0 = r30 - r40;
                float tmp13b1 = r31 - r41;

                float tmp00 = bias0 + r00 + tmp02a0 + tmp02b0;
                float tmp01 = bias1 + r01 + tmp02a1 + tmp02b1;
                float tmp10 = bias0 + tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                float tmp11 = bias1 + tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                float tmp20 = bias0 + tmp02a0 * 0.5f + tmp02b0 * 2;
                float tmp21 = bias1 + tmp02a1 * 0.5f + tmp02b1 * 2;
                float tmp30 = bias0 + r50 + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                float tmp31 = bias1 + r51 + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

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
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;
            const float* r4 = r0 + max_jj * 4;
            const float* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                float tmp02a = r1[0] + r2[0];
                float tmp02b = r3[0] + r4[0];
                float tmp13a = r1[0] - r2[0];
                float tmp13b = r3[0] - r4[0];

                tmp[0][m] = r0[0] + tmp02a + tmp02b;
                tmp[1][m] = tmp13a * sq2_d2 + tmp13b * sq2;
                tmp[2][m] = tmp02a * 0.5f + tmp02b * 2;
                tmp[3][m] = r5[0] + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float tmp02a = r1 + r2;
                float tmp02b = r3 + r4;
                float tmp13a = r1 - r2;
                float tmp13b = r3 - r4;

                float tmp0 = bias0 + r0 + tmp02a + tmp02b;
                float tmp1 = bias0 + tmp13a * sq2_d2 + tmp13b * sq2;
                float tmp2 = bias0 + tmp02a * 0.5f + tmp02b * 2;
                float tmp3 = bias0 + r5 + tmp13a * sq2_d4 + tmp13b * sq2_m2;

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

static void conv3x3s1_winograd43(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
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

    // NCNN_LOGE("conv3x3s1_winograd43 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd43_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd43_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
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

                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    float* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            // const float ktm[8][3] = {
            //     {1.0f, 0.0f, 0.0f},
            //     {-2.0f / 9, -2.0f / 9, -2.0f / 9},
            //     {-2.0f / 9, 2.0f / 9, -2.0f / 9},
            //     {1.0f / 90, 1.0f / 45, 2.0f / 45},
            //     {1.0f / 90, -1.0f / 45, 2.0f / 45},
            //     {1.0f / 45, 1.0f / 90, 1.0f / 180},
            //     {1.0f / 45, -1.0f / 90, 1.0f / 180},
            //     {0.0f, 0.0f, 1.0f}
            // };
            const float ktm0 = 2.0f / 9;
            const float ktm1 = 1.0f / 45;
            const float ktm2 = 2.0f / 45;
            const float ktm3 = 1.0f / 90;
            const float ktm4 = 1.0f / 180;

            float tmp[8][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = -r0 * ktm0 - r1 * ktm0 - r2 * ktm0;
                tmp[2][m] = -r0 * ktm0 + r1 * ktm0 - r2 * ktm0;
                tmp[3][m] = r0 * ktm3 + r1 * ktm1 + r2 * ktm2;
                tmp[4][m] = r0 * ktm3 - r1 * ktm1 + r2 * ktm2;
                tmp[5][m] = r0 * ktm1 + r1 * ktm3 + r2 * ktm4;
                tmp[6][m] = r0 * ktm1 - r1 * ktm3 + r2 * ktm4;
                tmp[7][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 8; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = -r0 * ktm0 - r1 * ktm0 - r2 * ktm0;
                float z2 = -r0 * ktm0 + r1 * ktm0 - r2 * ktm0;
                float z3 = r0 * ktm3 + r1 * ktm1 + r2 * ktm2;
                float z4 = r0 * ktm3 - r1 * ktm1 + r2 * ktm2;
                float z5 = r0 * ktm1 + r1 * ktm3 + r2 * ktm4;
                float z6 = r0 * ktm1 - r1 * ktm3 + r2 * ktm4;
                float z7 = r2;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp[4] = z4;
                ptmp[5] = z5;
                ptmp[6] = z6;
                ptmp[7] = z7;
                ptmp += 8;
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 64;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd63_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[8][8] = {
    //     {1.0f, 0.0f,-5.25f, 0.00f, 5.25f, 0.00f,-1.0f, 0.0f},
    //     {0.0f, 1.0f, 1.00f,-4.25f,-4.25f, 1.00f, 1.0f, 0.0f},
    //     {0.0f,-1.0f, 1.00f, 4.25f,-4.25f,-1.00f, 1.0f, 0.0f},
    //     {0.0f, 0.5f, 0.25f,-2.50f,-1.25f, 2.00f, 1.0f, 0.0f},
    //     {0.0f,-0.5f, 0.25f, 2.50f,-1.25f,-2.00f, 1.0f, 0.0f},
    //     {0.0f, 2.0f, 4.00f,-2.50f,-5.00f, 0.50f, 1.0f, 0.0f},
    //     {0.0f,-2.0f, 4.00f, 2.50f,-5.00f,-0.50f, 1.0f, 0.0f},
    //     {0.0f,-1.0f, 0.00f, 5.25f, 0.00f,-5.25f, 0.0f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 3) / 6;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __SSE2__
#if __AVX__
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
        float tmp[8][8][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                __m512 _r0 = _mm512_setzero_ps();
                __m512 _r1 = _mm512_setzero_ps();
                __m512 _r2 = _mm512_setzero_ps();
                __m512 _r3 = _mm512_setzero_ps();
                __m512 _r4 = _mm512_setzero_ps();
                __m512 _r5 = _mm512_setzero_ps();
                __m512 _r6 = _mm512_setzero_ps();
                __m512 _r7 = _mm512_setzero_ps();

                if (ti * 6 + m < h)
                {
                    if (elempack == 16)
                    {
                        _r0 = _mm512_load_ps(r0);
                        if (tj * 6 + 1 < w) _r1 = _mm512_load_ps(r0 + 16);
                        if (tj * 6 + 2 < w) _r2 = _mm512_load_ps(r0 + 32);
                        if (tj * 6 + 3 < w) _r3 = _mm512_load_ps(r0 + 48);
                        if (tj * 6 + 4 < w) _r4 = _mm512_load_ps(r0 + 64);
                        if (tj * 6 + 5 < w) _r5 = _mm512_load_ps(r0 + 80);
                        if (tj * 6 + 6 < w) _r6 = _mm512_load_ps(r0 + 96);
                        if (tj * 6 + 7 < w) _r7 = _mm512_load_ps(r0 + 112);
                    }
                    if (elempack == 8)
                    {
                        const float* r1 = r0 + N;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0)), _mm256_load_ps(r1), 1);
                        if (tj * 6 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 8)), _mm256_load_ps(r1 + 8), 1);
                        if (tj * 6 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 16)), _mm256_load_ps(r1 + 16), 1);
                        if (tj * 6 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 24)), _mm256_load_ps(r1 + 24), 1);
                        if (tj * 6 + 4 < w) _r4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 32)), _mm256_load_ps(r1 + 32), 1);
                        if (tj * 6 + 5 < w) _r5 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 40)), _mm256_load_ps(r1 + 40), 1);
                        if (tj * 6 + 6 < w) _r6 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 48)), _mm256_load_ps(r1 + 48), 1);
                        if (tj * 6 + 7 < w) _r7 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 56)), _mm256_load_ps(r1 + 56), 1);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2)), _mm_load_ps(r3), 1), 1);
                        if (tj * 6 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 4)), _mm_load_ps(r3 + 4), 1), 1);
                        if (tj * 6 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 8)), _mm_load_ps(r3 + 8), 1), 1);
                        if (tj * 6 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 12)), _mm_load_ps(r3 + 12), 1), 1);
                        if (tj * 6 + 4 < w) _r4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 16)), _mm_load_ps(r1 + 16), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 16)), _mm_load_ps(r3 + 16), 1), 1);
                        if (tj * 6 + 5 < w) _r5 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 20)), _mm_load_ps(r1 + 20), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 20)), _mm_load_ps(r3 + 20), 1), 1);
                        if (tj * 6 + 6 < w) _r6 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 24)), _mm_load_ps(r1 + 24), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 24)), _mm_load_ps(r3 + 24), 1), 1);
                        if (tj * 6 + 7 < w) _r7 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 28)), _mm_load_ps(r1 + 28), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 28)), _mm_load_ps(r3 + 28), 1), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;
                        const float* r8 = r0 + N * 8;
                        const float* r9 = r0 + N * 9;
                        const float* ra = r0 + N * 10;
                        const float* rb = r0 + N * 11;
                        const float* rc = r0 + N * 12;
                        const float* rd = r0 + N * 13;
                        const float* re = r0 + N * 14;
                        const float* rf = r0 + N * 15;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);
                        __m128 _t4 = _mm_loadu_ps(r4);
                        __m128 _t5 = _mm_loadu_ps(r5);
                        __m128 _t6 = _mm_loadu_ps(r6);
                        __m128 _t7 = _mm_loadu_ps(r7);
                        __m128 _t8 = _mm_loadu_ps(r8);
                        __m128 _t9 = _mm_loadu_ps(r9);
                        __m128 _ta = _mm_loadu_ps(ra);
                        __m128 _tb = _mm_loadu_ps(rb);
                        __m128 _tc = _mm_loadu_ps(rc);
                        __m128 _td = _mm_loadu_ps(rd);
                        __m128 _te = _mm_loadu_ps(re);
                        __m128 _tf = _mm_loadu_ps(rf);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);
                        _MM_TRANSPOSE4_PS(_t4, _t5, _t6, _t7);
                        _MM_TRANSPOSE4_PS(_t8, _t9, _ta, _tb);
                        _MM_TRANSPOSE4_PS(_tc, _td, _te, _tf);

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t0), _t4, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_t8), _tc, 1), 1);
                        if (tj * 6 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t1), _t5, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_t9), _td, 1), 1);
                        if (tj * 6 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t2), _t6, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_ta), _te, 1), 1);
                        if (tj * 6 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t3), _t7, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_tb), _tf, 1), 1);
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = _mm_loadu_ps(r0 + 4);
                            _t1 = _mm_loadu_ps(r1 + 4);
                            _t2 = _mm_loadu_ps(r2 + 4);
                            _t3 = _mm_loadu_ps(r3 + 4);
                            _t4 = _mm_loadu_ps(r4 + 4);
                            _t5 = _mm_loadu_ps(r5 + 4);
                            _t6 = _mm_loadu_ps(r6 + 4);
                            _t7 = _mm_loadu_ps(r7 + 4);
                            _t8 = _mm_loadu_ps(r8 + 4);
                            _t9 = _mm_loadu_ps(r9 + 4);
                            _ta = _mm_loadu_ps(ra + 4);
                            _tb = _mm_loadu_ps(rb + 4);
                            _tc = _mm_loadu_ps(rc + 4);
                            _td = _mm_loadu_ps(rd + 4);
                            _te = _mm_loadu_ps(re + 4);
                            _tf = _mm_loadu_ps(rf + 4);

                            _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);
                            _MM_TRANSPOSE4_PS(_t4, _t5, _t6, _t7);
                            _MM_TRANSPOSE4_PS(_t8, _t9, _ta, _tb);
                            _MM_TRANSPOSE4_PS(_tc, _td, _te, _tf);

                            _r4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t0), _t4, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_t8), _tc, 1), 1);
                            if (tj * 6 + 5 < w) _r5 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t1), _t5, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_t9), _td, 1), 1);
                            if (tj * 6 + 6 < w) _r6 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t2), _t6, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_ta), _te, 1), 1);
                            if (tj * 6 + 7 < w) _r7 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_t3), _t7, 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_tb), _tf, 1), 1);
                        }
                    }
                }

                __m512 _v5_25 = _mm512_set1_ps(5.25f);
                __m512 _vm4_25 = _mm512_set1_ps(-4.25f);
                __m512 _vm1_25 = _mm512_set1_ps(-1.25f);
                __m512 _v0_25 = _mm512_set1_ps(0.25f);
                __m512 _vm2_5 = _mm512_set1_ps(-2.5f);
                __m512 _v0_5 = _mm512_set1_ps(0.5f);
                __m512 _v2 = _mm512_set1_ps(2.f);
                __m512 _v4 = _mm512_set1_ps(4.f);

                __m512 _tmp12a = _mm512_fmadd_ps(_vm4_25, _r4, _mm512_add_ps(_r2, _r6));
                __m512 _tmp12b = _mm512_fmadd_ps(_vm4_25, _r3, _mm512_add_ps(_r1, _r5));
                __m512 _tmp34a = _mm512_fmadd_ps(_vm1_25, _r4, _mm512_fmadd_ps(_v0_25, _r2, _r6));
                __m512 _tmp34b = _mm512_fmadd_ps(_v2, _r5, _mm512_fmadd_ps(_vm2_5, _r3, _mm512_mul_ps(_r1, _v0_5)));
                __m512 _tmp56a = _mm512_fmadd_ps(_v4, _mm512_fmadd_ps(_vm1_25, _r4, _r2), _r6);
                __m512 _tmp56b = _mm512_fmadd_ps(_v0_5, _r5, _mm512_fmadd_ps(_vm2_5, _r3, _mm512_mul_ps(_r1, _v2)));

                __m512 _tmp0 = _mm512_fmadd_ps(_v5_25, _mm512_sub_ps(_r4, _r2), _mm512_sub_ps(_r0, _r6));
                __m512 _tmp1 = _mm512_add_ps(_tmp12a, _tmp12b);
                __m512 _tmp2 = _mm512_sub_ps(_tmp12a, _tmp12b);
                __m512 _tmp3 = _mm512_add_ps(_tmp34a, _tmp34b);
                __m512 _tmp4 = _mm512_sub_ps(_tmp34a, _tmp34b);
                __m512 _tmp5 = _mm512_add_ps(_tmp56a, _tmp56b);
                __m512 _tmp6 = _mm512_sub_ps(_tmp56a, _tmp56b);
                __m512 _tmp7 = _mm512_fmadd_ps(_v5_25, _mm512_sub_ps(_r3, _r5), _mm512_sub_ps(_r7, _r1));

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);
                _mm512_store_ps(tmp[2][m], _tmp2);
                _mm512_store_ps(tmp[3][m], _tmp3);
                _mm512_store_ps(tmp[4][m], _tmp4);
                _mm512_store_ps(tmp[5][m], _tmp5);
                _mm512_store_ps(tmp[6][m], _tmp6);
                _mm512_store_ps(tmp[7][m], _tmp7);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 16;
            float* p1 = p0 + max_jj * 16;
            float* p2 = p0 + max_jj * 16 * 2;
            float* p3 = p0 + max_jj * 16 * 3;
            float* p4 = p0 + max_jj * 16 * 4;
            float* p5 = p0 + max_jj * 16 * 5;
            float* p6 = p0 + max_jj * 16 * 6;
            float* p7 = p0 + max_jj * 16 * 7;

            for (int m = 0; m < 8; m++)
            {
                __m512 _r0 = _mm512_load_ps(tmp[m][0]);
                __m512 _r1 = _mm512_load_ps(tmp[m][1]);
                __m512 _r2 = _mm512_load_ps(tmp[m][2]);
                __m512 _r3 = _mm512_load_ps(tmp[m][3]);
                __m512 _r4 = _mm512_load_ps(tmp[m][4]);
                __m512 _r5 = _mm512_load_ps(tmp[m][5]);
                __m512 _r6 = _mm512_load_ps(tmp[m][6]);
                __m512 _r7 = _mm512_load_ps(tmp[m][7]);

                __m512 _v5_25 = _mm512_set1_ps(5.25f);
                __m512 _vm4_25 = _mm512_set1_ps(-4.25f);
                __m512 _vm1_25 = _mm512_set1_ps(-1.25f);
                __m512 _v0_25 = _mm512_set1_ps(0.25f);
                __m512 _vm2_5 = _mm512_set1_ps(-2.5f);
                __m512 _v0_5 = _mm512_set1_ps(0.5f);
                __m512 _v2 = _mm512_set1_ps(2.f);
                __m512 _v4 = _mm512_set1_ps(4.f);

                __m512 _tmp12a = _mm512_fmadd_ps(_vm4_25, _r4, _mm512_add_ps(_r2, _r6));
                __m512 _tmp12b = _mm512_fmadd_ps(_vm4_25, _r3, _mm512_add_ps(_r1, _r5));
                __m512 _tmp34a = _mm512_fmadd_ps(_vm1_25, _r4, _mm512_fmadd_ps(_v0_25, _r2, _r6));
                __m512 _tmp34b = _mm512_fmadd_ps(_v2, _r5, _mm512_fmadd_ps(_vm2_5, _r3, _mm512_mul_ps(_r1, _v0_5)));
                __m512 _tmp56a = _mm512_fmadd_ps(_v4, _mm512_fmadd_ps(_vm1_25, _r4, _r2), _r6);
                __m512 _tmp56b = _mm512_fmadd_ps(_v0_5, _r5, _mm512_fmadd_ps(_vm2_5, _r3, _mm512_mul_ps(_r1, _v2)));

                __m512 _tmp0 = _mm512_fmadd_ps(_v5_25, _mm512_sub_ps(_r4, _r2), _mm512_sub_ps(_r0, _r6));
                __m512 _tmp1 = _mm512_add_ps(_tmp12a, _tmp12b);
                __m512 _tmp2 = _mm512_sub_ps(_tmp12a, _tmp12b);
                __m512 _tmp3 = _mm512_add_ps(_tmp34a, _tmp34b);
                __m512 _tmp4 = _mm512_sub_ps(_tmp34a, _tmp34b);
                __m512 _tmp5 = _mm512_add_ps(_tmp56a, _tmp56b);
                __m512 _tmp6 = _mm512_sub_ps(_tmp56a, _tmp56b);
                __m512 _tmp7 = _mm512_fmadd_ps(_v5_25, _mm512_sub_ps(_r3, _r5), _mm512_sub_ps(_r7, _r1));

                _mm512_store_ps(p0, _tmp0);
                _mm512_store_ps(p1, _tmp1);
                _mm512_store_ps(p2, _tmp2);
                _mm512_store_ps(p3, _tmp3);
                _mm512_store_ps(p4, _tmp4);
                _mm512_store_ps(p5, _tmp5);
                _mm512_store_ps(p6, _tmp6);
                _mm512_store_ps(p7, _tmp7);

                p0 += max_jj * 8 * 16;
                p1 += max_jj * 8 * 16;
                p2 += max_jj * 8 * 16;
                p3 += max_jj * 8 * 16;
                p4 += max_jj * 8 * 16;
                p5 += max_jj * 8 * 16;
                p6 += max_jj * 8 * 16;
                p7 += max_jj * 8 * 16;
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
        float tmp[8][8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                __m256 _r0 = _mm256_setzero_ps();
                __m256 _r1 = _mm256_setzero_ps();
                __m256 _r2 = _mm256_setzero_ps();
                __m256 _r3 = _mm256_setzero_ps();
                __m256 _r4 = _mm256_setzero_ps();
                __m256 _r5 = _mm256_setzero_ps();
                __m256 _r6 = _mm256_setzero_ps();
                __m256 _r7 = _mm256_setzero_ps();

                if (ti * 6 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = _mm256_load_ps(r0);
                        if (tj * 6 + 1 < w) _r1 = _mm256_load_ps(r0 + 8);
                        if (tj * 6 + 2 < w) _r2 = _mm256_load_ps(r0 + 16);
                        if (tj * 6 + 3 < w) _r3 = _mm256_load_ps(r0 + 24);
                        if (tj * 6 + 4 < w) _r4 = _mm256_load_ps(r0 + 32);
                        if (tj * 6 + 5 < w) _r5 = _mm256_load_ps(r0 + 40);
                        if (tj * 6 + 6 < w) _r6 = _mm256_load_ps(r0 + 48);
                        if (tj * 6 + 7 < w) _r7 = _mm256_load_ps(r0 + 56);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1);
                        if (tj * 6 + 1 < w) _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1);
                        if (tj * 6 + 2 < w) _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1);
                        if (tj * 6 + 3 < w) _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1);
                        if (tj * 6 + 4 < w) _r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 16)), _mm_load_ps(r1 + 16), 1);
                        if (tj * 6 + 5 < w) _r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 20)), _mm_load_ps(r1 + 20), 1);
                        if (tj * 6 + 6 < w) _r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 24)), _mm_load_ps(r1 + 24), 1);
                        if (tj * 6 + 7 < w) _r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 28)), _mm_load_ps(r1 + 28), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);
                        __m128 _t4 = _mm_loadu_ps(r4);
                        __m128 _t5 = _mm_loadu_ps(r5);
                        __m128 _t6 = _mm_loadu_ps(r6);
                        __m128 _t7 = _mm_loadu_ps(r7);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);
                        _MM_TRANSPOSE4_PS(_t4, _t5, _t6, _t7);

                        _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t0), _t4, 1);
                        if (tj * 6 + 1 < w) _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t1), _t5, 1);
                        if (tj * 6 + 2 < w) _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t2), _t6, 1);
                        if (tj * 6 + 3 < w) _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t3), _t7, 1);
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = _mm_loadu_ps(r0 + 4);
                            _t1 = _mm_loadu_ps(r1 + 4);
                            _t2 = _mm_loadu_ps(r2 + 4);
                            _t3 = _mm_loadu_ps(r3 + 4);
                            _t4 = _mm_loadu_ps(r4 + 4);
                            _t5 = _mm_loadu_ps(r5 + 4);
                            _t6 = _mm_loadu_ps(r6 + 4);
                            _t7 = _mm_loadu_ps(r7 + 4);

                            _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);
                            _MM_TRANSPOSE4_PS(_t4, _t5, _t6, _t7);

                            _r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t0), _t4, 1);
                            if (tj * 6 + 5 < w) _r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t1), _t5, 1);
                            if (tj * 6 + 6 < w) _r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t2), _t6, 1);
                            if (tj * 6 + 7 < w) _r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_t3), _t7, 1);
                        }
                    }
                }

                __m256 _v5_25 = _mm256_set1_ps(5.25f);
                __m256 _vm4_25 = _mm256_set1_ps(-4.25f);
                __m256 _vm1_25 = _mm256_set1_ps(-1.25f);
                __m256 _v0_25 = _mm256_set1_ps(0.25f);
                __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
                __m256 _v0_5 = _mm256_set1_ps(0.5f);
                __m256 _v2 = _mm256_set1_ps(2.f);
                __m256 _v4 = _mm256_set1_ps(4.f);

                __m256 _tmp12a = _mm256_comp_fmadd_ps(_vm4_25, _r4, _mm256_add_ps(_r2, _r6));
                __m256 _tmp12b = _mm256_comp_fmadd_ps(_vm4_25, _r3, _mm256_add_ps(_r1, _r5));
                __m256 _tmp34a = _mm256_comp_fmadd_ps(_vm1_25, _r4, _mm256_comp_fmadd_ps(_v0_25, _r2, _r6));
                __m256 _tmp34b = _mm256_comp_fmadd_ps(_v2, _r5, _mm256_comp_fmadd_ps(_vm2_5, _r3, _mm256_mul_ps(_r1, _v0_5)));
                __m256 _tmp56a = _mm256_comp_fmadd_ps(_v4, _mm256_comp_fmadd_ps(_vm1_25, _r4, _r2), _r6);
                __m256 _tmp56b = _mm256_comp_fmadd_ps(_v0_5, _r5, _mm256_comp_fmadd_ps(_vm2_5, _r3, _mm256_mul_ps(_r1, _v2)));

                __m256 _tmp0 = _mm256_comp_fmadd_ps(_v5_25, _mm256_sub_ps(_r4, _r2), _mm256_sub_ps(_r0, _r6));
                __m256 _tmp1 = _mm256_add_ps(_tmp12a, _tmp12b);
                __m256 _tmp2 = _mm256_sub_ps(_tmp12a, _tmp12b);
                __m256 _tmp3 = _mm256_add_ps(_tmp34a, _tmp34b);
                __m256 _tmp4 = _mm256_sub_ps(_tmp34a, _tmp34b);
                __m256 _tmp5 = _mm256_add_ps(_tmp56a, _tmp56b);
                __m256 _tmp6 = _mm256_sub_ps(_tmp56a, _tmp56b);
                __m256 _tmp7 = _mm256_comp_fmadd_ps(_v5_25, _mm256_sub_ps(_r3, _r5), _mm256_sub_ps(_r7, _r1));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm256_storeu_ps(tmp[0][m], _tmp0);
                _mm256_storeu_ps(tmp[1][m], _tmp1);
                _mm256_storeu_ps(tmp[2][m], _tmp2);
                _mm256_storeu_ps(tmp[3][m], _tmp3);
                _mm256_storeu_ps(tmp[4][m], _tmp4);
                _mm256_storeu_ps(tmp[5][m], _tmp5);
                _mm256_storeu_ps(tmp[6][m], _tmp6);
                _mm256_storeu_ps(tmp[7][m], _tmp7);
#else
                _mm256_store_ps(tmp[0][m], _tmp0);
                _mm256_store_ps(tmp[1][m], _tmp1);
                _mm256_store_ps(tmp[2][m], _tmp2);
                _mm256_store_ps(tmp[3][m], _tmp3);
                _mm256_store_ps(tmp[4][m], _tmp4);
                _mm256_store_ps(tmp[5][m], _tmp5);
                _mm256_store_ps(tmp[6][m], _tmp6);
                _mm256_store_ps(tmp[7][m], _tmp7);
#endif

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;
            float* p4 = p0 + max_jj * 8 * 4;
            float* p5 = p0 + max_jj * 8 * 5;
            float* p6 = p0 + max_jj * 8 * 6;
            float* p7 = p0 + max_jj * 8 * 7;

            for (int m = 0; m < 8; m++)
            {
#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m256 _r0 = _mm256_loadu_ps(tmp[m][0]);
                __m256 _r1 = _mm256_loadu_ps(tmp[m][1]);
                __m256 _r2 = _mm256_loadu_ps(tmp[m][2]);
                __m256 _r3 = _mm256_loadu_ps(tmp[m][3]);
                __m256 _r4 = _mm256_loadu_ps(tmp[m][4]);
                __m256 _r5 = _mm256_loadu_ps(tmp[m][5]);
                __m256 _r6 = _mm256_loadu_ps(tmp[m][6]);
                __m256 _r7 = _mm256_loadu_ps(tmp[m][7]);
#else
                __m256 _r0 = _mm256_load_ps(tmp[m][0]);
                __m256 _r1 = _mm256_load_ps(tmp[m][1]);
                __m256 _r2 = _mm256_load_ps(tmp[m][2]);
                __m256 _r3 = _mm256_load_ps(tmp[m][3]);
                __m256 _r4 = _mm256_load_ps(tmp[m][4]);
                __m256 _r5 = _mm256_load_ps(tmp[m][5]);
                __m256 _r6 = _mm256_load_ps(tmp[m][6]);
                __m256 _r7 = _mm256_load_ps(tmp[m][7]);
#endif

                __m256 _v5_25 = _mm256_set1_ps(5.25f);
                __m256 _vm4_25 = _mm256_set1_ps(-4.25f);
                __m256 _vm1_25 = _mm256_set1_ps(-1.25f);
                __m256 _v0_25 = _mm256_set1_ps(0.25f);
                __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
                __m256 _v0_5 = _mm256_set1_ps(0.5f);
                __m256 _v2 = _mm256_set1_ps(2.f);
                __m256 _v4 = _mm256_set1_ps(4.f);

                __m256 _tmp12a = _mm256_comp_fmadd_ps(_vm4_25, _r4, _mm256_add_ps(_r2, _r6));
                __m256 _tmp12b = _mm256_comp_fmadd_ps(_vm4_25, _r3, _mm256_add_ps(_r1, _r5));
                __m256 _tmp34a = _mm256_comp_fmadd_ps(_vm1_25, _r4, _mm256_comp_fmadd_ps(_v0_25, _r2, _r6));
                __m256 _tmp34b = _mm256_comp_fmadd_ps(_v2, _r5, _mm256_comp_fmadd_ps(_vm2_5, _r3, _mm256_mul_ps(_r1, _v0_5)));
                __m256 _tmp56a = _mm256_comp_fmadd_ps(_v4, _mm256_comp_fmadd_ps(_vm1_25, _r4, _r2), _r6);
                __m256 _tmp56b = _mm256_comp_fmadd_ps(_v0_5, _r5, _mm256_comp_fmadd_ps(_vm2_5, _r3, _mm256_mul_ps(_r1, _v2)));

                __m256 _tmp0 = _mm256_comp_fmadd_ps(_v5_25, _mm256_sub_ps(_r4, _r2), _mm256_sub_ps(_r0, _r6));
                __m256 _tmp1 = _mm256_add_ps(_tmp12a, _tmp12b);
                __m256 _tmp2 = _mm256_sub_ps(_tmp12a, _tmp12b);
                __m256 _tmp3 = _mm256_add_ps(_tmp34a, _tmp34b);
                __m256 _tmp4 = _mm256_sub_ps(_tmp34a, _tmp34b);
                __m256 _tmp5 = _mm256_add_ps(_tmp56a, _tmp56b);
                __m256 _tmp6 = _mm256_sub_ps(_tmp56a, _tmp56b);
                __m256 _tmp7 = _mm256_comp_fmadd_ps(_v5_25, _mm256_sub_ps(_r3, _r5), _mm256_sub_ps(_r7, _r1));

                _mm256_store_ps(p0, _tmp0);
                _mm256_store_ps(p1, _tmp1);
                _mm256_store_ps(p2, _tmp2);
                _mm256_store_ps(p3, _tmp3);
                _mm256_store_ps(p4, _tmp4);
                _mm256_store_ps(p5, _tmp5);
                _mm256_store_ps(p6, _tmp6);
                _mm256_store_ps(p7, _tmp7);

                p0 += max_jj * 8 * 8;
                p1 += max_jj * 8 * 8;
                p2 += max_jj * 8 * 8;
                p3 += max_jj * 8 * 8;
                p4 += max_jj * 8 * 8;
                p5 += max_jj * 8 * 8;
                p6 += max_jj * 8 * 8;
                p7 += max_jj * 8 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
#else // __AVX__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __AVX__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[8][8][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                __m128 _r0 = _mm_setzero_ps();
                __m128 _r1 = _mm_setzero_ps();
                __m128 _r2 = _mm_setzero_ps();
                __m128 _r3 = _mm_setzero_ps();
                __m128 _r4 = _mm_setzero_ps();
                __m128 _r5 = _mm_setzero_ps();
                __m128 _r6 = _mm_setzero_ps();
                __m128 _r7 = _mm_setzero_ps();

                if (ti * 6 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = _mm_load_ps(r0);
                        if (tj * 6 + 1 < w) _r1 = _mm_load_ps(r0 + 4);
                        if (tj * 6 + 2 < w) _r2 = _mm_load_ps(r0 + 8);
                        if (tj * 6 + 3 < w) _r3 = _mm_load_ps(r0 + 12);
                        if (tj * 6 + 4 < w) _r4 = _mm_load_ps(r0 + 16);
                        if (tj * 6 + 5 < w) _r5 = _mm_load_ps(r0 + 20);
                        if (tj * 6 + 6 < w) _r6 = _mm_load_ps(r0 + 24);
                        if (tj * 6 + 7 < w) _r7 = _mm_load_ps(r0 + 28);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 6 + 1 < w) _r1 = _t1;
                        if (tj * 6 + 2 < w) _r2 = _t2;
                        if (tj * 6 + 3 < w) _r3 = _t3;
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = _mm_loadu_ps(r0 + 4);
                            _t1 = _mm_loadu_ps(r1 + 4);
                            _t2 = _mm_loadu_ps(r2 + 4);
                            _t3 = _mm_loadu_ps(r3 + 4);

                            _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);

                            _r4 = _t0;
                            if (tj * 6 + 5 < w) _r5 = _t1;
                            if (tj * 6 + 6 < w) _r6 = _t2;
                            if (tj * 6 + 7 < w) _r7 = _t3;
                        }
                    }
                }

                __m128 _v5_25 = _mm_set1_ps(5.25f);
                __m128 _vm4_25 = _mm_set1_ps(-4.25f);
                __m128 _vm1_25 = _mm_set1_ps(-1.25f);
                __m128 _v0_25 = _mm_set1_ps(0.25f);
                __m128 _vm2_5 = _mm_set1_ps(-2.5f);
                __m128 _v0_5 = _mm_set1_ps(0.5f);
                __m128 _v2 = _mm_set1_ps(2.f);
                __m128 _v4 = _mm_set1_ps(4.f);

                __m128 _tmp12a = _mm_comp_fmadd_ps(_vm4_25, _r4, _mm_add_ps(_r2, _r6));
                __m128 _tmp12b = _mm_comp_fmadd_ps(_vm4_25, _r3, _mm_add_ps(_r1, _r5));
                __m128 _tmp34a = _mm_comp_fmadd_ps(_vm1_25, _r4, _mm_comp_fmadd_ps(_v0_25, _r2, _r6));
                __m128 _tmp34b = _mm_comp_fmadd_ps(_v2, _r5, _mm_comp_fmadd_ps(_vm2_5, _r3, _mm_mul_ps(_r1, _v0_5)));
                __m128 _tmp56a = _mm_comp_fmadd_ps(_v4, _mm_comp_fmadd_ps(_vm1_25, _r4, _r2), _r6);
                __m128 _tmp56b = _mm_comp_fmadd_ps(_v0_5, _r5, _mm_comp_fmadd_ps(_vm2_5, _r3, _mm_mul_ps(_r1, _v2)));

                __m128 _tmp0 = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_r4, _r2), _mm_sub_ps(_r0, _r6));
                __m128 _tmp1 = _mm_add_ps(_tmp12a, _tmp12b);
                __m128 _tmp2 = _mm_sub_ps(_tmp12a, _tmp12b);
                __m128 _tmp3 = _mm_add_ps(_tmp34a, _tmp34b);
                __m128 _tmp4 = _mm_sub_ps(_tmp34a, _tmp34b);
                __m128 _tmp5 = _mm_add_ps(_tmp56a, _tmp56b);
                __m128 _tmp6 = _mm_sub_ps(_tmp56a, _tmp56b);
                __m128 _tmp7 = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_r3, _r5), _mm_sub_ps(_r7, _r1));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_ps(tmp[0][m], _tmp0);
                _mm_storeu_ps(tmp[1][m], _tmp1);
                _mm_storeu_ps(tmp[2][m], _tmp2);
                _mm_storeu_ps(tmp[3][m], _tmp3);
                _mm_storeu_ps(tmp[4][m], _tmp4);
                _mm_storeu_ps(tmp[5][m], _tmp5);
                _mm_storeu_ps(tmp[6][m], _tmp6);
                _mm_storeu_ps(tmp[7][m], _tmp7);
#else
                _mm_store_ps(tmp[0][m], _tmp0);
                _mm_store_ps(tmp[1][m], _tmp1);
                _mm_store_ps(tmp[2][m], _tmp2);
                _mm_store_ps(tmp[3][m], _tmp3);
                _mm_store_ps(tmp[4][m], _tmp4);
                _mm_store_ps(tmp[5][m], _tmp5);
                _mm_store_ps(tmp[6][m], _tmp6);
                _mm_store_ps(tmp[7][m], _tmp7);
#endif

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;
            float* p4 = p0 + max_jj * 4 * 4;
            float* p5 = p0 + max_jj * 4 * 5;
            float* p6 = p0 + max_jj * 4 * 6;
            float* p7 = p0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128 _r0 = _mm_loadu_ps(tmp[m][0]);
                __m128 _r1 = _mm_loadu_ps(tmp[m][1]);
                __m128 _r2 = _mm_loadu_ps(tmp[m][2]);
                __m128 _r3 = _mm_loadu_ps(tmp[m][3]);
                __m128 _r4 = _mm_loadu_ps(tmp[m][4]);
                __m128 _r5 = _mm_loadu_ps(tmp[m][5]);
                __m128 _r6 = _mm_loadu_ps(tmp[m][6]);
                __m128 _r7 = _mm_loadu_ps(tmp[m][7]);
#else
                __m128 _r0 = _mm_load_ps(tmp[m][0]);
                __m128 _r1 = _mm_load_ps(tmp[m][1]);
                __m128 _r2 = _mm_load_ps(tmp[m][2]);
                __m128 _r3 = _mm_load_ps(tmp[m][3]);
                __m128 _r4 = _mm_load_ps(tmp[m][4]);
                __m128 _r5 = _mm_load_ps(tmp[m][5]);
                __m128 _r6 = _mm_load_ps(tmp[m][6]);
                __m128 _r7 = _mm_load_ps(tmp[m][7]);
#endif

                __m128 _v5_25 = _mm_set1_ps(5.25f);
                __m128 _vm4_25 = _mm_set1_ps(-4.25f);
                __m128 _vm1_25 = _mm_set1_ps(-1.25f);
                __m128 _v0_25 = _mm_set1_ps(0.25f);
                __m128 _vm2_5 = _mm_set1_ps(-2.5f);
                __m128 _v0_5 = _mm_set1_ps(0.5f);
                __m128 _v2 = _mm_set1_ps(2.f);
                __m128 _v4 = _mm_set1_ps(4.f);

                __m128 _tmp12a = _mm_comp_fmadd_ps(_vm4_25, _r4, _mm_add_ps(_r2, _r6));
                __m128 _tmp12b = _mm_comp_fmadd_ps(_vm4_25, _r3, _mm_add_ps(_r1, _r5));
                __m128 _tmp34a = _mm_comp_fmadd_ps(_vm1_25, _r4, _mm_comp_fmadd_ps(_v0_25, _r2, _r6));
                __m128 _tmp34b = _mm_comp_fmadd_ps(_v2, _r5, _mm_comp_fmadd_ps(_vm2_5, _r3, _mm_mul_ps(_r1, _v0_5)));
                __m128 _tmp56a = _mm_comp_fmadd_ps(_v4, _mm_comp_fmadd_ps(_vm1_25, _r4, _r2), _r6);
                __m128 _tmp56b = _mm_comp_fmadd_ps(_v0_5, _r5, _mm_comp_fmadd_ps(_vm2_5, _r3, _mm_mul_ps(_r1, _v2)));

                __m128 _tmp0 = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_r4, _r2), _mm_sub_ps(_r0, _r6));
                __m128 _tmp1 = _mm_add_ps(_tmp12a, _tmp12b);
                __m128 _tmp2 = _mm_sub_ps(_tmp12a, _tmp12b);
                __m128 _tmp3 = _mm_add_ps(_tmp34a, _tmp34b);
                __m128 _tmp4 = _mm_sub_ps(_tmp34a, _tmp34b);
                __m128 _tmp5 = _mm_add_ps(_tmp56a, _tmp56b);
                __m128 _tmp6 = _mm_sub_ps(_tmp56a, _tmp56b);
                __m128 _tmp7 = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_r3, _r5), _mm_sub_ps(_r7, _r1));

                _mm_store_ps(p0, _tmp0);
                _mm_store_ps(p1, _tmp1);
                _mm_store_ps(p2, _tmp2);
                _mm_store_ps(p3, _tmp3);
                _mm_store_ps(p4, _tmp4);
                _mm_store_ps(p5, _tmp5);
                _mm_store_ps(p6, _tmp6);
                _mm_store_ps(p7, _tmp7);

                p0 += max_jj * 8 * 4;
                p1 += max_jj * 8 * 4;
                p2 += max_jj * 8 * 4;
                p3 += max_jj * 8 * 4;
                p4 += max_jj * 8 * 4;
                p5 += max_jj * 8 * 4;
                p6 += max_jj * 8 * 4;
                p7 += max_jj * 8 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __SSE2__
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __SSE2__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        float tmp[8][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
                float r40 = 0.f;
                float r41 = 0.f;
                float r50 = 0.f;
                float r51 = 0.f;
                float r60 = 0.f;
                float r61 = 0.f;
                float r70 = 0.f;
                float r71 = 0.f;

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 6 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 6 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 6 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                        if (tj * 6 + 4 < w)
                        {
                            r40 = r0[4];
                            r41 = r1[4];
                        }
                        if (tj * 6 + 5 < w)
                        {
                            r50 = r0[5];
                            r51 = r1[5];
                        }
                        if (tj * 6 + 6 < w)
                        {
                            r60 = r0[6];
                            r61 = r1[6];
                        }
                        if (tj * 6 + 7 < w)
                        {
                            r70 = r0[7];
                            r71 = r1[7];
                        }
                    }
                }

                float tmp12a0 = r20 + r60 - r40 * 4.25f;
                float tmp12a1 = r21 + r61 - r41 * 4.25f;
                float tmp12b0 = r10 + r50 - r30 * 4.25f;
                float tmp12b1 = r11 + r51 - r31 * 4.25f;
                float tmp34a0 = r60 + r20 * 0.25f - r40 * 1.25f;
                float tmp34a1 = r61 + r21 * 0.25f - r41 * 1.25f;
                float tmp34b0 = r10 * 0.5f - r30 * 2.5f + r50 * 2.f;
                float tmp34b1 = r11 * 0.5f - r31 * 2.5f + r51 * 2.f;
                float tmp56a0 = r20 * 4.f - r40 * 5.f + r60;
                float tmp56a1 = r21 * 4.f - r41 * 5.f + r61;
                float tmp56b0 = r10 * 2.f - r30 * 2.5f + r50 * 0.5f;
                float tmp56b1 = r11 * 2.f - r31 * 2.5f + r51 * 0.5f;

                tmp[0][m][0] = r00 - r60 + (r40 - r20) * 5.25f;
                tmp[0][m][1] = r01 - r61 + (r41 - r21) * 5.25f;
                tmp[1][m][0] = tmp12a0 + tmp12b0;
                tmp[1][m][1] = tmp12a1 + tmp12b1;
                tmp[2][m][0] = tmp12a0 - tmp12b0;
                tmp[2][m][1] = tmp12a1 - tmp12b1;
                tmp[3][m][0] = tmp34a0 + tmp34b0;
                tmp[3][m][1] = tmp34a1 + tmp34b1;
                tmp[4][m][0] = tmp34a0 - tmp34b0;
                tmp[4][m][1] = tmp34a1 - tmp34b1;
                tmp[5][m][0] = tmp56a0 + tmp56b0;
                tmp[5][m][1] = tmp56a1 + tmp56b1;
                tmp[6][m][0] = tmp56a0 - tmp56b0;
                tmp[6][m][1] = tmp56a1 - tmp56b1;
                tmp[7][m][0] = r70 - r10 + (r30 - r50) * 5.25f;
                tmp[7][m][1] = r71 - r11 + (r31 - r51) * 5.25f;

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;
            float* p4 = p0 + max_jj * 2 * 4;
            float* p5 = p0 + max_jj * 2 * 5;
            float* p6 = p0 + max_jj * 2 * 6;
            float* p7 = p0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];
                float r60 = tmp[m][6][0];
                float r61 = tmp[m][6][1];
                float r70 = tmp[m][7][0];
                float r71 = tmp[m][7][1];

                float tmp12a0 = r20 + r60 - r40 * 4.25f;
                float tmp12a1 = r21 + r61 - r41 * 4.25f;
                float tmp12b0 = r10 + r50 - r30 * 4.25f;
                float tmp12b1 = r11 + r51 - r31 * 4.25f;
                float tmp34a0 = r60 + r20 * 0.25f - r40 * 1.25f;
                float tmp34a1 = r61 + r21 * 0.25f - r41 * 1.25f;
                float tmp34b0 = r10 * 0.5f - r30 * 2.5f + r50 * 2.f;
                float tmp34b1 = r11 * 0.5f - r31 * 2.5f + r51 * 2.f;
                float tmp56a0 = r20 * 4.f - r40 * 5.f + r60;
                float tmp56a1 = r21 * 4.f - r41 * 5.f + r61;
                float tmp56b0 = r10 * 2.f - r30 * 2.5f + r50 * 0.5f;
                float tmp56b1 = r11 * 2.f - r31 * 2.5f + r51 * 0.5f;

                p0[0] = r00 - r60 + (r40 - r20) * 5.25f;
                p0[1] = r01 - r61 + (r41 - r21) * 5.25f;
                p1[0] = tmp12a0 + tmp12b0;
                p1[1] = tmp12a1 + tmp12b1;
                p2[0] = tmp12a0 - tmp12b0;
                p2[1] = tmp12a1 - tmp12b1;
                p3[0] = tmp34a0 + tmp34b0;
                p3[1] = tmp34a1 + tmp34b1;
                p4[0] = tmp34a0 - tmp34b0;
                p4[1] = tmp34a1 - tmp34b1;
                p5[0] = tmp56a0 + tmp56b0;
                p5[1] = tmp56a1 + tmp56b1;
                p6[0] = tmp56a0 - tmp56b0;
                p6[1] = tmp56a1 - tmp56b1;
                p7[0] = r70 - r10 + (r30 - r50) * 5.25f;
                p7[1] = r71 - r11 + (r31 - r51) * 5.25f;

                p0 += max_jj * 8 * 2;
                p1 += max_jj * 8 * 2;
                p2 += max_jj * 8 * 2;
                p3 += max_jj * 8 * 2;
                p4 += max_jj * 8 * 2;
                p5 += max_jj * 8 * 2;
                p6 += max_jj * 8 * 2;
                p7 += max_jj * 8 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        float tmp[8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;
                float r4 = 0.f;
                float r5 = 0.f;
                float r6 = 0.f;
                float r7 = 0.f;

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 6 + 1 < w) r1 = r0123[1];
                        if (tj * 6 + 2 < w) r2 = r0123[2];
                        if (tj * 6 + 3 < w) r3 = r0123[3];
                        if (tj * 6 + 4 < w) r4 = r0123[4];
                        if (tj * 6 + 5 < w) r5 = r0123[5];
                        if (tj * 6 + 6 < w) r6 = r0123[6];
                        if (tj * 6 + 7 < w) r7 = r0123[7];
                    }
                }

                float tmp12a = r2 + r6 - r4 * 4.25f;
                float tmp12b = r1 + r5 - r3 * 4.25f;
                float tmp34a = r6 + r2 * 0.25f - r4 * 1.25f;
                float tmp34b = r1 * 0.5f - r3 * 2.5f + r5 * 2.f;
                float tmp56a = r2 * 4.f - r4 * 5.f + r6;
                float tmp56b = r1 * 2.f - r3 * 2.5f + r5 * 0.5f;

                tmp[0][m] = r0 - r6 + (r4 - r2) * 5.25f;
                tmp[1][m] = tmp12a + tmp12b;
                tmp[2][m] = tmp12a - tmp12b;
                tmp[3][m] = tmp34a + tmp34b;
                tmp[4][m] = tmp34a - tmp34b;
                tmp[5][m] = tmp56a + tmp56b;
                tmp[6][m] = tmp56a - tmp56b;
                tmp[7][m] = r7 - r1 + (r3 - r5) * 5.25f;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;
            float* p4 = p0 + max_jj * 4;
            float* p5 = p0 + max_jj * 5;
            float* p6 = p0 + max_jj * 6;
            float* p7 = p0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];
                float r6 = tmp[m][6];
                float r7 = tmp[m][7];

                float tmp12a = r2 + r6 - r4 * 4.25f;
                float tmp12b = r1 + r5 - r3 * 4.25f;
                float tmp34a = r6 + r2 * 0.25f - r4 * 1.25f;
                float tmp34b = r1 * 0.5f - r3 * 2.5f + r5 * 2.f;
                float tmp56a = r2 * 4.f - r4 * 5.f + r6;
                float tmp56b = r1 * 2.f - r3 * 2.5f + r5 * 0.5f;

                p0[0] = r0 - r6 + (r4 - r2) * 5.25f;
                p1[0] = tmp12a + tmp12b;
                p2[0] = tmp12a - tmp12b;
                p3[0] = tmp34a + tmp34b;
                p4[0] = tmp34a - tmp34b;
                p5[0] = tmp56a + tmp56b;
                p6[0] = tmp56a - tmp56b;
                p7[0] = r7 - r1 + (r3 - r5) * 5.25f;

                p0 += max_jj * 8;
                p1 += max_jj * 8;
                p2 += max_jj * 8;
                p3 += max_jj * 8;
                p4 += max_jj * 8;
                p5 += max_jj * 8;
                p6 += max_jj * 8;
                p7 += max_jj * 8;
            }
        }
    }
}

static inline void conv3x3s1_winograd63_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[6][8] = {
    //     {1.0f, 1.0f,  1.0f,  1.0f,  1.0f, 32.0f, 32.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  2.0f, -2.0f, 16.0f,-16.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f,  4.0f,  4.0f,  8.0f,  8.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  8.0f, -8.0f,  4.0f, -4.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f, 16.0f, 16.0f,  2.0f,  2.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 32.0f,-32.0f,  1.0f, -1.0f, 1.0f}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 5) / 6;

    const float* biasptr = bias;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        __m512 _bias0 = biasptr ? _mm512_loadu_ps(biasptr + i + ii) : _mm512_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[6][8][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 16;
            const float* r1 = r0 + max_jj * 16;
            const float* r2 = r0 + max_jj * 16 * 2;
            const float* r3 = r0 + max_jj * 16 * 3;
            const float* r4 = r0 + max_jj * 16 * 4;
            const float* r5 = r0 + max_jj * 16 * 5;
            const float* r6 = r0 + max_jj * 16 * 6;
            const float* r7 = r0 + max_jj * 16 * 7;

            __m512 _v32 = _mm512_set1_ps(32.f);
            __m512 _v16 = _mm512_set1_ps(16.f);
            __m512 _v8 = _mm512_set1_ps(8.f);
            __m512 _v4 = _mm512_set1_ps(4.f);
            __m512 _v2 = _mm512_set1_ps(2.f);

            for (int m = 0; m < 8; m++)
            {
                __m512 _r0 = _mm512_load_ps(r0);
                __m512 _r1 = _mm512_load_ps(r1);
                __m512 _r2 = _mm512_load_ps(r2);
                __m512 _r3 = _mm512_load_ps(r3);
                __m512 _r4 = _mm512_load_ps(r4);
                __m512 _r5 = _mm512_load_ps(r5);
                __m512 _r6 = _mm512_load_ps(r6);
                __m512 _r7 = _mm512_load_ps(r7);

                __m512 _tmp024a = _mm512_add_ps(_r1, _r2);
                __m512 _tmp135a = _mm512_sub_ps(_r1, _r2);
                __m512 _tmp024b = _mm512_add_ps(_r3, _r4);
                __m512 _tmp135b = _mm512_sub_ps(_r3, _r4);
                __m512 _tmp024c = _mm512_add_ps(_r5, _r6);
                __m512 _tmp135c = _mm512_sub_ps(_r5, _r6);
                __m512 _tmp0 = _mm512_add_ps(_mm512_add_ps(_r0, _tmp024a), _mm512_fmadd_ps(_v32, _tmp024c, _tmp024b));
                __m512 _tmp1 = _mm512_fmadd_ps(_v16, _tmp135c, _mm512_fmadd_ps(_v2, _tmp135b, _tmp135a));
                __m512 _tmp2 = _mm512_fmadd_ps(_v8, _tmp024c, _mm512_fmadd_ps(_v4, _tmp024b, _tmp024a));
                __m512 _tmp3 = _mm512_fmadd_ps(_v4, _tmp135c, _mm512_fmadd_ps(_v8, _tmp135b, _tmp135a));
                __m512 _tmp4 = _mm512_fmadd_ps(_v2, _tmp024c, _mm512_fmadd_ps(_v16, _tmp024b, _tmp024a));
                __m512 _tmp5 = _mm512_add_ps(_mm512_add_ps(_r7, _tmp135a), _mm512_fmadd_ps(_v32, _tmp135b, _tmp135c));

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);
                _mm512_store_ps(tmp[2][m], _tmp2);
                _mm512_store_ps(tmp[3][m], _tmp3);
                _mm512_store_ps(tmp[4][m], _tmp4);
                _mm512_store_ps(tmp[5][m], _tmp5);

                r0 += max_jj * 8 * 16;
                r1 += max_jj * 8 * 16;
                r2 += max_jj * 8 * 16;
                r3 += max_jj * 8 * 16;
                r4 += max_jj * 8 * 16;
                r5 += max_jj * 8 * 16;
                r6 += max_jj * 8 * 16;
                r7 += max_jj * 8 * 16;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                __m512 _r0 = _mm512_load_ps(tmp[m][0]);
                __m512 _r1 = _mm512_load_ps(tmp[m][1]);
                __m512 _r2 = _mm512_load_ps(tmp[m][2]);
                __m512 _r3 = _mm512_load_ps(tmp[m][3]);
                __m512 _r4 = _mm512_load_ps(tmp[m][4]);
                __m512 _r5 = _mm512_load_ps(tmp[m][5]);
                __m512 _r6 = _mm512_load_ps(tmp[m][6]);
                __m512 _r7 = _mm512_load_ps(tmp[m][7]);

                __m512 _tmp024a = _mm512_add_ps(_r1, _r2);
                __m512 _tmp135a = _mm512_sub_ps(_r1, _r2);
                __m512 _tmp024b = _mm512_add_ps(_r3, _r4);
                __m512 _tmp135b = _mm512_sub_ps(_r3, _r4);
                __m512 _tmp024c = _mm512_add_ps(_r5, _r6);
                __m512 _tmp135c = _mm512_sub_ps(_r5, _r6);
                __m512 _tmp0 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_add_ps(_r0, _tmp024a), _mm512_fmadd_ps(_v32, _tmp024c, _tmp024b)));
                __m512 _tmp1 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_v16, _tmp135c, _mm512_fmadd_ps(_v2, _tmp135b, _tmp135a)));
                __m512 _tmp2 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_v8, _tmp024c, _mm512_fmadd_ps(_v4, _tmp024b, _tmp024a)));
                __m512 _tmp3 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_v4, _tmp135c, _mm512_fmadd_ps(_v8, _tmp135b, _tmp135a)));
                __m512 _tmp4 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_v2, _tmp024c, _mm512_fmadd_ps(_v16, _tmp024b, _tmp024a)));
                __m512 _tmp5 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_add_ps(_r7, _tmp135a), _mm512_fmadd_ps(_v32, _tmp135b, _tmp135c)));

                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _tmp0);
                    if (tj * 6 + 1 < outw) _mm512_store_ps(outptr0 + 16, _tmp1);
                    if (tj * 6 + 2 < outw) _mm512_store_ps(outptr0 + 32, _tmp2);
                    if (tj * 6 + 3 < outw) _mm512_store_ps(outptr0 + 48, _tmp3);
                    if (tj * 6 + 4 < outw) _mm512_store_ps(outptr0 + 64, _tmp4);
                    if (tj * 6 + 5 < outw) _mm512_store_ps(outptr0 + 80, _tmp5);
                }
                if (out_elempack == 8)
                {
                    float* outptr1 = outptr0 + N;

                    _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_tmp0, 0));
                    _mm256_store_ps(outptr1, _mm512_extractf32x8_ps(_tmp0, 1));
                    if (tj * 6 + 1 < outw)
                    {
                        _mm256_store_ps(outptr0 + 8, _mm512_extractf32x8_ps(_tmp1, 0));
                        _mm256_store_ps(outptr1 + 8, _mm512_extractf32x8_ps(_tmp1, 1));
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        _mm256_store_ps(outptr0 + 16, _mm512_extractf32x8_ps(_tmp2, 0));
                        _mm256_store_ps(outptr1 + 16, _mm512_extractf32x8_ps(_tmp2, 1));
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        _mm256_store_ps(outptr0 + 24, _mm512_extractf32x8_ps(_tmp3, 0));
                        _mm256_store_ps(outptr1 + 24, _mm512_extractf32x8_ps(_tmp3, 1));
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        _mm256_store_ps(outptr0 + 32, _mm512_extractf32x8_ps(_tmp4, 0));
                        _mm256_store_ps(outptr1 + 32, _mm512_extractf32x8_ps(_tmp4, 1));
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        _mm256_store_ps(outptr0 + 40, _mm512_extractf32x8_ps(_tmp5, 0));
                        _mm256_store_ps(outptr1 + 40, _mm512_extractf32x8_ps(_tmp5, 1));
                    }
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_tmp0, 0));
                    _mm_store_ps(outptr1, _mm512_extractf32x4_ps(_tmp0, 1));
                    _mm_store_ps(outptr2, _mm512_extractf32x4_ps(_tmp0, 2));
                    _mm_store_ps(outptr3, _mm512_extractf32x4_ps(_tmp0, 3));
                    if (tj * 6 + 1 < outw)
                    {
                        _mm_store_ps(outptr0 + 4, _mm512_extractf32x4_ps(_tmp1, 0));
                        _mm_store_ps(outptr1 + 4, _mm512_extractf32x4_ps(_tmp1, 1));
                        _mm_store_ps(outptr2 + 4, _mm512_extractf32x4_ps(_tmp1, 2));
                        _mm_store_ps(outptr3 + 4, _mm512_extractf32x4_ps(_tmp1, 3));
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        _mm_store_ps(outptr0 + 8, _mm512_extractf32x4_ps(_tmp2, 0));
                        _mm_store_ps(outptr1 + 8, _mm512_extractf32x4_ps(_tmp2, 1));
                        _mm_store_ps(outptr2 + 8, _mm512_extractf32x4_ps(_tmp2, 2));
                        _mm_store_ps(outptr3 + 8, _mm512_extractf32x4_ps(_tmp2, 3));
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        _mm_store_ps(outptr0 + 12, _mm512_extractf32x4_ps(_tmp3, 0));
                        _mm_store_ps(outptr1 + 12, _mm512_extractf32x4_ps(_tmp3, 1));
                        _mm_store_ps(outptr2 + 12, _mm512_extractf32x4_ps(_tmp3, 2));
                        _mm_store_ps(outptr3 + 12, _mm512_extractf32x4_ps(_tmp3, 3));
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        _mm_store_ps(outptr0 + 16, _mm512_extractf32x4_ps(_tmp4, 0));
                        _mm_store_ps(outptr1 + 16, _mm512_extractf32x4_ps(_tmp4, 1));
                        _mm_store_ps(outptr2 + 16, _mm512_extractf32x4_ps(_tmp4, 2));
                        _mm_store_ps(outptr3 + 16, _mm512_extractf32x4_ps(_tmp4, 3));
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        _mm_store_ps(outptr0 + 20, _mm512_extractf32x4_ps(_tmp5, 0));
                        _mm_store_ps(outptr1 + 20, _mm512_extractf32x4_ps(_tmp5, 1));
                        _mm_store_ps(outptr2 + 20, _mm512_extractf32x4_ps(_tmp5, 2));
                        _mm_store_ps(outptr3 + 20, _mm512_extractf32x4_ps(_tmp5, 3));
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[16];
                    float tmp1[16];
                    float tmp2[16];
                    float tmp3[16];
                    float tmp4[16];
                    float tmp5[16];
                    _mm512_storeu_ps(tmp0, _tmp0);
                    _mm512_storeu_ps(tmp1, _tmp1);
                    _mm512_storeu_ps(tmp2, _tmp2);
                    _mm512_storeu_ps(tmp3, _tmp3);
                    _mm512_storeu_ps(tmp4, _tmp4);
                    _mm512_storeu_ps(tmp5, _tmp5);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;
                    float* outptr8 = outptr0 + N * 8;
                    float* outptr9 = outptr0 + N * 9;
                    float* outptra = outptr0 + N * 10;
                    float* outptrb = outptr0 + N * 11;
                    float* outptrc = outptr0 + N * 12;
                    float* outptrd = outptr0 + N * 13;
                    float* outptre = outptr0 + N * 14;
                    float* outptrf = outptr0 + N * 15;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    outptr8[0] = tmp0[8];
                    outptr9[0] = tmp0[9];
                    outptra[0] = tmp0[10];
                    outptrb[0] = tmp0[11];
                    outptrc[0] = tmp0[12];
                    outptrd[0] = tmp0[13];
                    outptre[0] = tmp0[14];
                    outptrf[0] = tmp0[15];
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                        outptr8[1] = tmp1[8];
                        outptr9[1] = tmp1[9];
                        outptra[1] = tmp1[10];
                        outptrb[1] = tmp1[11];
                        outptrc[1] = tmp1[12];
                        outptrd[1] = tmp1[13];
                        outptre[1] = tmp1[14];
                        outptrf[1] = tmp1[15];
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                        outptr4[2] = tmp2[4];
                        outptr5[2] = tmp2[5];
                        outptr6[2] = tmp2[6];
                        outptr7[2] = tmp2[7];
                        outptr8[2] = tmp2[8];
                        outptr9[2] = tmp2[9];
                        outptra[2] = tmp2[10];
                        outptrb[2] = tmp2[11];
                        outptrc[2] = tmp2[12];
                        outptrd[2] = tmp2[13];
                        outptre[2] = tmp2[14];
                        outptrf[2] = tmp2[15];
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                        outptr4[3] = tmp3[4];
                        outptr5[3] = tmp3[5];
                        outptr6[3] = tmp3[6];
                        outptr7[3] = tmp3[7];
                        outptr8[3] = tmp3[8];
                        outptr9[3] = tmp3[9];
                        outptra[3] = tmp3[10];
                        outptrb[3] = tmp3[11];
                        outptrc[3] = tmp3[12];
                        outptrd[3] = tmp3[13];
                        outptre[3] = tmp3[14];
                        outptrf[3] = tmp3[15];
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                        outptr4[4] = tmp4[4];
                        outptr5[4] = tmp4[5];
                        outptr6[4] = tmp4[6];
                        outptr7[4] = tmp4[7];
                        outptr8[4] = tmp4[8];
                        outptr9[4] = tmp4[9];
                        outptra[4] = tmp4[10];
                        outptrb[4] = tmp4[11];
                        outptrc[4] = tmp4[12];
                        outptrd[4] = tmp4[13];
                        outptre[4] = tmp4[14];
                        outptrf[4] = tmp4[15];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                        outptr4[5] = tmp5[4];
                        outptr5[5] = tmp5[5];
                        outptr6[5] = tmp5[6];
                        outptr7[5] = tmp5[7];
                        outptr8[5] = tmp5[8];
                        outptr9[5] = tmp5[9];
                        outptra[5] = tmp5[10];
                        outptrb[5] = tmp5[11];
                        outptrc[5] = tmp5[12];
                        outptrd[5] = tmp5[13];
                        outptre[5] = tmp5[14];
                        outptrf[5] = tmp5[15];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _bias0 = biasptr ? _mm256_loadu_ps(biasptr + i + ii) : _mm256_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(32))
#else
        __attribute__((aligned(32)))
#endif
        float tmp[6][8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;
            const float* r4 = r0 + max_jj * 8 * 4;
            const float* r5 = r0 + max_jj * 8 * 5;
            const float* r6 = r0 + max_jj * 8 * 6;
            const float* r7 = r0 + max_jj * 8 * 7;

            __m256 _v32 = _mm256_set1_ps(32.f);
            __m256 _v16 = _mm256_set1_ps(16.f);
            __m256 _v8 = _mm256_set1_ps(8.f);
            __m256 _v4 = _mm256_set1_ps(4.f);
            __m256 _v2 = _mm256_set1_ps(2.f);

            for (int m = 0; m < 8; m++)
            {
                __m256 _r0 = _mm256_load_ps(r0);
                __m256 _r1 = _mm256_load_ps(r1);
                __m256 _r2 = _mm256_load_ps(r2);
                __m256 _r3 = _mm256_load_ps(r3);
                __m256 _r4 = _mm256_load_ps(r4);
                __m256 _r5 = _mm256_load_ps(r5);
                __m256 _r6 = _mm256_load_ps(r6);
                __m256 _r7 = _mm256_load_ps(r7);

                __m256 _tmp024a = _mm256_add_ps(_r1, _r2);
                __m256 _tmp135a = _mm256_sub_ps(_r1, _r2);
                __m256 _tmp024b = _mm256_add_ps(_r3, _r4);
                __m256 _tmp135b = _mm256_sub_ps(_r3, _r4);
                __m256 _tmp024c = _mm256_add_ps(_r5, _r6);
                __m256 _tmp135c = _mm256_sub_ps(_r5, _r6);
                __m256 _tmp0 = _mm256_add_ps(_mm256_add_ps(_r0, _tmp024a), _mm256_comp_fmadd_ps(_v32, _tmp024c, _tmp024b));
                __m256 _tmp1 = _mm256_comp_fmadd_ps(_v16, _tmp135c, _mm256_comp_fmadd_ps(_v2, _tmp135b, _tmp135a));
                __m256 _tmp2 = _mm256_comp_fmadd_ps(_v8, _tmp024c, _mm256_comp_fmadd_ps(_v4, _tmp024b, _tmp024a));
                __m256 _tmp3 = _mm256_comp_fmadd_ps(_v4, _tmp135c, _mm256_comp_fmadd_ps(_v8, _tmp135b, _tmp135a));
                __m256 _tmp4 = _mm256_comp_fmadd_ps(_v2, _tmp024c, _mm256_comp_fmadd_ps(_v16, _tmp024b, _tmp024a));
                __m256 _tmp5 = _mm256_add_ps(_mm256_add_ps(_r7, _tmp135a), _mm256_comp_fmadd_ps(_v32, _tmp135b, _tmp135c));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm256_storeu_ps(tmp[0][m], _tmp0);
                _mm256_storeu_ps(tmp[1][m], _tmp1);
                _mm256_storeu_ps(tmp[2][m], _tmp2);
                _mm256_storeu_ps(tmp[3][m], _tmp3);
                _mm256_storeu_ps(tmp[4][m], _tmp4);
                _mm256_storeu_ps(tmp[5][m], _tmp5);
#else
                _mm256_store_ps(tmp[0][m], _tmp0);
                _mm256_store_ps(tmp[1][m], _tmp1);
                _mm256_store_ps(tmp[2][m], _tmp2);
                _mm256_store_ps(tmp[3][m], _tmp3);
                _mm256_store_ps(tmp[4][m], _tmp4);
                _mm256_store_ps(tmp[5][m], _tmp5);
#endif

                r0 += max_jj * 8 * 8;
                r1 += max_jj * 8 * 8;
                r2 += max_jj * 8 * 8;
                r3 += max_jj * 8 * 8;
                r4 += max_jj * 8 * 8;
                r5 += max_jj * 8 * 8;
                r6 += max_jj * 8 * 8;
                r7 += max_jj * 8 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m256 _r0 = _mm256_loadu_ps(tmp[m][0]);
                __m256 _r1 = _mm256_loadu_ps(tmp[m][1]);
                __m256 _r2 = _mm256_loadu_ps(tmp[m][2]);
                __m256 _r3 = _mm256_loadu_ps(tmp[m][3]);
                __m256 _r4 = _mm256_loadu_ps(tmp[m][4]);
                __m256 _r5 = _mm256_loadu_ps(tmp[m][5]);
                __m256 _r6 = _mm256_loadu_ps(tmp[m][6]);
                __m256 _r7 = _mm256_loadu_ps(tmp[m][7]);
#else
                __m256 _r0 = _mm256_load_ps(tmp[m][0]);
                __m256 _r1 = _mm256_load_ps(tmp[m][1]);
                __m256 _r2 = _mm256_load_ps(tmp[m][2]);
                __m256 _r3 = _mm256_load_ps(tmp[m][3]);
                __m256 _r4 = _mm256_load_ps(tmp[m][4]);
                __m256 _r5 = _mm256_load_ps(tmp[m][5]);
                __m256 _r6 = _mm256_load_ps(tmp[m][6]);
                __m256 _r7 = _mm256_load_ps(tmp[m][7]);
#endif

                __m256 _tmp024a = _mm256_add_ps(_r1, _r2);
                __m256 _tmp135a = _mm256_sub_ps(_r1, _r2);
                __m256 _tmp024b = _mm256_add_ps(_r3, _r4);
                __m256 _tmp135b = _mm256_sub_ps(_r3, _r4);
                __m256 _tmp024c = _mm256_add_ps(_r5, _r6);
                __m256 _tmp135c = _mm256_sub_ps(_r5, _r6);
                __m256 _tmp0 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_r0, _tmp024a), _mm256_comp_fmadd_ps(_v32, _tmp024c, _tmp024b)));
                __m256 _tmp1 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_v16, _tmp135c, _mm256_comp_fmadd_ps(_v2, _tmp135b, _tmp135a)));
                __m256 _tmp2 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_v8, _tmp024c, _mm256_comp_fmadd_ps(_v4, _tmp024b, _tmp024a)));
                __m256 _tmp3 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_v4, _tmp135c, _mm256_comp_fmadd_ps(_v8, _tmp135b, _tmp135a)));
                __m256 _tmp4 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_v2, _tmp024c, _mm256_comp_fmadd_ps(_v16, _tmp024b, _tmp024a)));
                __m256 _tmp5 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_r7, _tmp135a), _mm256_comp_fmadd_ps(_v32, _tmp135b, _tmp135c)));

                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _tmp0);
                    if (tj * 6 + 1 < outw) _mm256_store_ps(outptr0 + 8, _tmp1);
                    if (tj * 6 + 2 < outw) _mm256_store_ps(outptr0 + 16, _tmp2);
                    if (tj * 6 + 3 < outw) _mm256_store_ps(outptr0 + 24, _tmp3);
                    if (tj * 6 + 4 < outw) _mm256_store_ps(outptr0 + 32, _tmp4);
                    if (tj * 6 + 5 < outw) _mm256_store_ps(outptr0 + 40, _tmp5);
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    _mm_store_ps(outptr0, _mm256_extractf128_ps(_tmp0, 0));
                    _mm_store_ps(outptr1, _mm256_extractf128_ps(_tmp0, 1));
                    if (tj * 6 + 1 < outw)
                    {
                        _mm_store_ps(outptr0 + 4, _mm256_extractf128_ps(_tmp1, 0));
                        _mm_store_ps(outptr1 + 4, _mm256_extractf128_ps(_tmp1, 1));
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        _mm_store_ps(outptr0 + 8, _mm256_extractf128_ps(_tmp2, 0));
                        _mm_store_ps(outptr1 + 8, _mm256_extractf128_ps(_tmp2, 1));
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        _mm_store_ps(outptr0 + 12, _mm256_extractf128_ps(_tmp3, 0));
                        _mm_store_ps(outptr1 + 12, _mm256_extractf128_ps(_tmp3, 1));
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        _mm_store_ps(outptr0 + 16, _mm256_extractf128_ps(_tmp4, 0));
                        _mm_store_ps(outptr1 + 16, _mm256_extractf128_ps(_tmp4, 1));
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        _mm_store_ps(outptr0 + 20, _mm256_extractf128_ps(_tmp5, 0));
                        _mm_store_ps(outptr1 + 20, _mm256_extractf128_ps(_tmp5, 1));
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    float tmp2[8];
                    float tmp3[8];
                    float tmp4[8];
                    float tmp5[8];
                    _mm256_storeu_ps(tmp0, _tmp0);
                    _mm256_storeu_ps(tmp1, _tmp1);
                    _mm256_storeu_ps(tmp2, _tmp2);
                    _mm256_storeu_ps(tmp3, _tmp3);
                    _mm256_storeu_ps(tmp4, _tmp4);
                    _mm256_storeu_ps(tmp5, _tmp5);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 6 + 1 < outw)
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
                    if (tj * 6 + 2 < outw)
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
                    if (tj * 6 + 3 < outw)
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
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                        outptr4[4] = tmp4[4];
                        outptr5[4] = tmp4[5];
                        outptr6[4] = tmp4[6];
                        outptr7[4] = tmp4[7];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                        outptr4[5] = tmp5[4];
                        outptr5[5] = tmp5[5];
                        outptr6[5] = tmp5[6];
                        outptr7[5] = tmp5[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _bias0 = biasptr ? _mm_loadu_ps(biasptr + i + ii) : _mm_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][8][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;
            const float* r4 = r0 + max_jj * 4 * 4;
            const float* r5 = r0 + max_jj * 4 * 5;
            const float* r6 = r0 + max_jj * 4 * 6;
            const float* r7 = r0 + max_jj * 4 * 7;

            __m128 _v32 = _mm_set1_ps(32.f);
            __m128 _v16 = _mm_set1_ps(16.f);
            __m128 _v8 = _mm_set1_ps(8.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _v2 = _mm_set1_ps(2.f);

            for (int m = 0; m < 8; m++)
            {
                __m128 _r0 = _mm_load_ps(r0);
                __m128 _r1 = _mm_load_ps(r1);
                __m128 _r2 = _mm_load_ps(r2);
                __m128 _r3 = _mm_load_ps(r3);
                __m128 _r4 = _mm_load_ps(r4);
                __m128 _r5 = _mm_load_ps(r5);
                __m128 _r6 = _mm_load_ps(r6);
                __m128 _r7 = _mm_load_ps(r7);

                __m128 _tmp024a = _mm_add_ps(_r1, _r2);
                __m128 _tmp135a = _mm_sub_ps(_r1, _r2);
                __m128 _tmp024b = _mm_add_ps(_r3, _r4);
                __m128 _tmp135b = _mm_sub_ps(_r3, _r4);
                __m128 _tmp024c = _mm_add_ps(_r5, _r6);
                __m128 _tmp135c = _mm_sub_ps(_r5, _r6);
                __m128 _tmp0 = _mm_add_ps(_mm_add_ps(_r0, _tmp024a), _mm_comp_fmadd_ps(_v32, _tmp024c, _tmp024b));
                __m128 _tmp1 = _mm_comp_fmadd_ps(_v16, _tmp135c, _mm_comp_fmadd_ps(_v2, _tmp135b, _tmp135a));
                __m128 _tmp2 = _mm_comp_fmadd_ps(_v8, _tmp024c, _mm_comp_fmadd_ps(_v4, _tmp024b, _tmp024a));
                __m128 _tmp3 = _mm_comp_fmadd_ps(_v4, _tmp135c, _mm_comp_fmadd_ps(_v8, _tmp135b, _tmp135a));
                __m128 _tmp4 = _mm_comp_fmadd_ps(_v2, _tmp024c, _mm_comp_fmadd_ps(_v16, _tmp024b, _tmp024a));
                __m128 _tmp5 = _mm_add_ps(_mm_add_ps(_r7, _tmp135a), _mm_comp_fmadd_ps(_v32, _tmp135b, _tmp135c));

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                _mm_storeu_ps(tmp[0][m], _tmp0);
                _mm_storeu_ps(tmp[1][m], _tmp1);
                _mm_storeu_ps(tmp[2][m], _tmp2);
                _mm_storeu_ps(tmp[3][m], _tmp3);
                _mm_storeu_ps(tmp[4][m], _tmp4);
                _mm_storeu_ps(tmp[5][m], _tmp5);
#else
                _mm_store_ps(tmp[0][m], _tmp0);
                _mm_store_ps(tmp[1][m], _tmp1);
                _mm_store_ps(tmp[2][m], _tmp2);
                _mm_store_ps(tmp[3][m], _tmp3);
                _mm_store_ps(tmp[4][m], _tmp4);
                _mm_store_ps(tmp[5][m], _tmp5);
#endif

                r0 += max_jj * 8 * 4;
                r1 += max_jj * 8 * 4;
                r2 += max_jj * 8 * 4;
                r3 += max_jj * 8 * 4;
                r4 += max_jj * 8 * 4;
                r5 += max_jj * 8 * 4;
                r6 += max_jj * 8 * 4;
                r7 += max_jj * 8 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                __m128 _r0 = _mm_loadu_ps(tmp[m][0]);
                __m128 _r1 = _mm_loadu_ps(tmp[m][1]);
                __m128 _r2 = _mm_loadu_ps(tmp[m][2]);
                __m128 _r3 = _mm_loadu_ps(tmp[m][3]);
                __m128 _r4 = _mm_loadu_ps(tmp[m][4]);
                __m128 _r5 = _mm_loadu_ps(tmp[m][5]);
                __m128 _r6 = _mm_loadu_ps(tmp[m][6]);
                __m128 _r7 = _mm_loadu_ps(tmp[m][7]);
#else
                __m128 _r0 = _mm_load_ps(tmp[m][0]);
                __m128 _r1 = _mm_load_ps(tmp[m][1]);
                __m128 _r2 = _mm_load_ps(tmp[m][2]);
                __m128 _r3 = _mm_load_ps(tmp[m][3]);
                __m128 _r4 = _mm_load_ps(tmp[m][4]);
                __m128 _r5 = _mm_load_ps(tmp[m][5]);
                __m128 _r6 = _mm_load_ps(tmp[m][6]);
                __m128 _r7 = _mm_load_ps(tmp[m][7]);
#endif

                __m128 _tmp024a = _mm_add_ps(_r1, _r2);
                __m128 _tmp135a = _mm_sub_ps(_r1, _r2);
                __m128 _tmp024b = _mm_add_ps(_r3, _r4);
                __m128 _tmp135b = _mm_sub_ps(_r3, _r4);
                __m128 _tmp024c = _mm_add_ps(_r5, _r6);
                __m128 _tmp135c = _mm_sub_ps(_r5, _r6);
                __m128 _tmp0 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_r0, _tmp024a), _mm_comp_fmadd_ps(_v32, _tmp024c, _tmp024b)));
                __m128 _tmp1 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v16, _tmp135c, _mm_comp_fmadd_ps(_v2, _tmp135b, _tmp135a)));
                __m128 _tmp2 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v8, _tmp024c, _mm_comp_fmadd_ps(_v4, _tmp024b, _tmp024a)));
                __m128 _tmp3 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v4, _tmp135c, _mm_comp_fmadd_ps(_v8, _tmp135b, _tmp135a)));
                __m128 _tmp4 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v2, _tmp024c, _mm_comp_fmadd_ps(_v16, _tmp024b, _tmp024a)));
                __m128 _tmp5 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_r7, _tmp135a), _mm_comp_fmadd_ps(_v32, _tmp135b, _tmp135c)));

                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _tmp0);
                    if (tj * 6 + 1 < outw) _mm_store_ps(outptr0 + 4, _tmp1);
                    if (tj * 6 + 2 < outw) _mm_store_ps(outptr0 + 8, _tmp2);
                    if (tj * 6 + 3 < outw) _mm_store_ps(outptr0 + 12, _tmp3);
                    if (tj * 6 + 4 < outw) _mm_store_ps(outptr0 + 16, _tmp4);
                    if (tj * 6 + 5 < outw) _mm_store_ps(outptr0 + 20, _tmp5);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    float tmp2[4];
                    float tmp3[4];
                    float tmp4[4];
                    float tmp5[4];
                    _mm_storeu_ps(tmp0, _tmp0);
                    _mm_storeu_ps(tmp1, _tmp1);
                    _mm_storeu_ps(tmp2, _tmp2);
                    _mm_storeu_ps(tmp3, _tmp3);
                    _mm_storeu_ps(tmp4, _tmp4);
                    _mm_storeu_ps(tmp5, _tmp5);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

        float tmp[6][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;
            const float* r4 = r0 + max_jj * 2 * 4;
            const float* r5 = r0 + max_jj * 2 * 5;
            const float* r6 = r0 + max_jj * 2 * 6;
            const float* r7 = r0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
                float tmp024a0 = r1[0] + r2[0];
                float tmp024a1 = r1[1] + r2[1];
                float tmp135a0 = r1[0] - r2[0];
                float tmp135a1 = r1[1] - r2[1];
                float tmp024b0 = r3[0] + r4[0];
                float tmp024b1 = r3[1] + r4[1];
                float tmp135b0 = r3[0] - r4[0];
                float tmp135b1 = r3[1] - r4[1];
                float tmp024c0 = r5[0] + r6[0];
                float tmp024c1 = r5[1] + r6[1];
                float tmp135c0 = r5[0] - r6[0];
                float tmp135c1 = r5[1] - r6[1];

                tmp[0][m][0] = r0[0] + tmp024a0 + tmp024b0 + tmp024c0 * 32;
                tmp[0][m][1] = r0[1] + tmp024a1 + tmp024b1 + tmp024c1 * 32;
                tmp[1][m][0] = tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * 16;
                tmp[1][m][1] = tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * 16;
                tmp[2][m][0] = tmp024a0 + tmp024b0 * 4 + tmp024c0 * 8;
                tmp[2][m][1] = tmp024a1 + tmp024b1 * 4 + tmp024c1 * 8;
                tmp[3][m][0] = tmp135a0 + tmp135b0 * 8 + tmp135c0 * 4;
                tmp[3][m][1] = tmp135a1 + tmp135b1 * 8 + tmp135c1 * 4;
                tmp[4][m][0] = tmp024a0 + tmp024b0 * 16 + tmp024c0 + tmp024c0;
                tmp[4][m][1] = tmp024a1 + tmp024b1 * 16 + tmp024c1 + tmp024c1;
                tmp[5][m][0] = r7[0] + tmp135a0 + tmp135b0 * 32 + tmp135c0;
                tmp[5][m][1] = r7[1] + tmp135a1 + tmp135b1 * 32 + tmp135c1;

                r0 += max_jj * 8 * 2;
                r1 += max_jj * 8 * 2;
                r2 += max_jj * 8 * 2;
                r3 += max_jj * 8 * 2;
                r4 += max_jj * 8 * 2;
                r5 += max_jj * 8 * 2;
                r6 += max_jj * 8 * 2;
                r7 += max_jj * 8 * 2;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];
                float r60 = tmp[m][6][0];
                float r61 = tmp[m][6][1];
                float r70 = tmp[m][7][0];
                float r71 = tmp[m][7][1];

                float tmp024a0 = r10 + r20;
                float tmp024a1 = r11 + r21;
                float tmp135a0 = r10 - r20;
                float tmp135a1 = r11 - r21;
                float tmp024b0 = r30 + r40;
                float tmp024b1 = r31 + r41;
                float tmp135b0 = r30 - r40;
                float tmp135b1 = r31 - r41;
                float tmp024c0 = r50 + r60;
                float tmp024c1 = r51 + r61;
                float tmp135c0 = r50 - r60;
                float tmp135c1 = r51 - r61;

                float tmp00 = bias0 + r00 + tmp024a0 + tmp024b0 + tmp024c0 * 32;
                float tmp01 = bias1 + r01 + tmp024a1 + tmp024b1 + tmp024c1 * 32;
                float tmp10 = bias0 + tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * 16;
                float tmp11 = bias1 + tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * 16;
                float tmp20 = bias0 + tmp024a0 + tmp024b0 * 4 + tmp024c0 * 8;
                float tmp21 = bias1 + tmp024a1 + tmp024b1 * 4 + tmp024c1 * 8;
                float tmp30 = bias0 + tmp135a0 + tmp135b0 * 8 + tmp135c0 * 4;
                float tmp31 = bias1 + tmp135a1 + tmp135b1 * 8 + tmp135c1 * 4;
                float tmp40 = bias0 + tmp024a0 + tmp024b0 * 16 + tmp024c0 + tmp024c0;
                float tmp41 = bias1 + tmp024a1 + tmp024b1 * 16 + tmp024c1 + tmp024c1;
                float tmp50 = bias0 + r70 + tmp135a0 + tmp135b0 * 32 + tmp135c0;
                float tmp51 = bias1 + r71 + tmp135a1 + tmp135b1 * 32 + tmp135c1;

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp20;
                        outptr1[2] = tmp21;
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp30;
                        outptr1[3] = tmp31;
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp40;
                        outptr1[4] = tmp41;
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp50;
                        outptr1[5] = tmp51;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;
            const float* r4 = r0 + max_jj * 4;
            const float* r5 = r0 + max_jj * 5;
            const float* r6 = r0 + max_jj * 6;
            const float* r7 = r0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                float tmp024a = r1[0] + r2[0];
                float tmp135a = r1[0] - r2[0];
                float tmp024b = r3[0] + r4[0];
                float tmp135b = r3[0] - r4[0];
                float tmp024c = r5[0] + r6[0];
                float tmp135c = r5[0] - r6[0];

                tmp[0][m] = r0[0] + tmp024a + tmp024b + tmp024c * 32;
                tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;
                tmp[5][m] = r7[0] + tmp135a + tmp135b * 32 + tmp135c;

                r0 += max_jj * 8;
                r1 += max_jj * 8;
                r2 += max_jj * 8;
                r3 += max_jj * 8;
                r4 += max_jj * 8;
                r5 += max_jj * 8;
                r6 += max_jj * 8;
                r7 += max_jj * 8;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];
                float r6 = tmp[m][6];
                float r7 = tmp[m][7];

                float tmp024a = r1 + r2;
                float tmp135a = r1 - r2;
                float tmp024b = r3 + r4;
                float tmp135b = r3 - r4;
                float tmp024c = r5 + r6;
                float tmp135c = r5 - r6;

                float tmp0 = bias0 + r0 + tmp024a + tmp024b + tmp024c * 32;
                float tmp1 = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                float tmp2 = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                float tmp3 = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                float tmp4 = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;
                float tmp5 = bias0 + r7 + tmp135a + tmp135b * 32 + tmp135c;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 6 + 1 < outw) outptr0[1] = tmp1;
                    if (tj * 6 + 2 < outw) outptr0[2] = tmp2;
                    if (tj * 6 + 3 < outw) outptr0[3] = tmp3;
                    if (tj * 6 + 4 < outw) outptr0[4] = tmp4;
                    if (tj * 6 + 5 < outw) outptr0[5] = tmp5;
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd63(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 6n+2, winograd F(6,3)
    int w_tiles = (outw + 5) / 6;
    int h_tiles = (outh + 5) / 6;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 64;

    // NCNN_LOGE("conv3x3s1_winograd63 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd63_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd63_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
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

                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd63_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

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
    for (int b = 0; b < batch; b++)
    {
        float* pp = AT.row(b);

        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            const float* p0 = A;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[ii * max_kk * batch + b];
                pp[1] = p0[(ii + 1) * max_kk * batch + b];
                pp[2] = p0[(ii + 2) * max_kk * batch + b];
                pp[3] = p0[(ii + 3) * max_kk * batch + b];
                pp[4] = p0[(ii + 4) * max_kk * batch + b];
                pp[5] = p0[(ii + 5) * max_kk * batch + b];
                pp[6] = p0[(ii + 6) * max_kk * batch + b];
                pp[7] = p0[(ii + 7) * max_kk * batch + b];
                pp[8] = p0[(ii + 8) * max_kk * batch + b];
                pp[9] = p0[(ii + 9) * max_kk * batch + b];
                pp[10] = p0[(ii + 10) * max_kk * batch + b];
                pp[11] = p0[(ii + 11) * max_kk * batch + b];
                pp[12] = p0[(ii + 12) * max_kk * batch + b];
                pp[13] = p0[(ii + 13) * max_kk * batch + b];
                pp[14] = p0[(ii + 14) * max_kk * batch + b];
                pp[15] = p0[(ii + 15) * max_kk * batch + b];
                p0 += batch;
                pp += 16;
            }
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = A;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[ii * max_kk * batch + b];
                pp[1] = p0[(ii + 1) * max_kk * batch + b];
                pp[2] = p0[(ii + 2) * max_kk * batch + b];
                pp[3] = p0[(ii + 3) * max_kk * batch + b];
                pp[4] = p0[(ii + 4) * max_kk * batch + b];
                pp[5] = p0[(ii + 5) * max_kk * batch + b];
                pp[6] = p0[(ii + 6) * max_kk * batch + b];
                pp[7] = p0[(ii + 7) * max_kk * batch + b];
                p0 += batch;
                pp += 8;
            }
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = A;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[ii * max_kk * batch + b];
                pp[1] = p0[(ii + 1) * max_kk * batch + b];
                pp[2] = p0[(ii + 2) * max_kk * batch + b];
                pp[3] = p0[(ii + 3) * max_kk * batch + b];
                p0 += batch;
                pp += 4;
            }
        }
#endif // __SSE2__
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = A;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[ii * max_kk * batch + b];
                pp[1] = p0[(ii + 1) * max_kk * batch + b];
                p0 += batch;
                pp += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const float* p0 = A;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[ii * max_kk * batch + b];
                p0 += batch;
                pp += 1;
            }
        }
    }
}

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk)
{
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
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0 + jj * batch * 16 + b * 16);
                __m512 _r1 = _mm512_load_ps(p0 + (jj + 1) * batch * 16 + b * 16);
                __m512 _r2 = _mm512_load_ps(p0 + (jj + 2) * batch * 16 + b * 16);
                __m512 _r3 = _mm512_load_ps(p0 + (jj + 3) * batch * 16 + b * 16);
                __m512 _r4 = _mm512_load_ps(p0 + (jj + 4) * batch * 16 + b * 16);
                __m512 _r5 = _mm512_load_ps(p0 + (jj + 5) * batch * 16 + b * 16);
                __m512 _r6 = _mm512_load_ps(p0 + (jj + 6) * batch * 16 + b * 16);
                __m512 _r7 = _mm512_load_ps(p0 + (jj + 7) * batch * 16 + b * 16);
                __m512 _r8 = _mm512_load_ps(p0 + (jj + 8) * batch * 16 + b * 16);
                __m512 _r9 = _mm512_load_ps(p0 + (jj + 9) * batch * 16 + b * 16);
                __m512 _ra = _mm512_load_ps(p0 + (jj + 10) * batch * 16 + b * 16);
                __m512 _rb = _mm512_load_ps(p0 + (jj + 11) * batch * 16 + b * 16);
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
#endif // __AVX512F__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0 + jj * batch * 8 + b * 8);
                __m256 _r1 = _mm256_load_ps(p0 + (jj + 1) * batch * 8 + b * 8);
                __m256 _r2 = _mm256_load_ps(p0 + (jj + 2) * batch * 8 + b * 8);
                __m256 _r3 = _mm256_load_ps(p0 + (jj + 3) * batch * 8 + b * 8);
                __m256 _r4 = _mm256_load_ps(p0 + (jj + 4) * batch * 8 + b * 8);
                __m256 _r5 = _mm256_load_ps(p0 + (jj + 5) * batch * 8 + b * 8);
                __m256 _r6 = _mm256_load_ps(p0 + (jj + 6) * batch * 8 + b * 8);
                __m256 _r7 = _mm256_load_ps(p0 + (jj + 7) * batch * 8 + b * 8);
                __m256 _r8 = _mm256_load_ps(p0 + (jj + 8) * batch * 8 + b * 8);
                __m256 _r9 = _mm256_load_ps(p0 + (jj + 9) * batch * 8 + b * 8);
                __m256 _ra = _mm256_load_ps(p0 + (jj + 10) * batch * 8 + b * 8);
                __m256 _rb = _mm256_load_ps(p0 + (jj + 11) * batch * 8 + b * 8);
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
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0 + jj * batch * 4 + b * 4);
                __m128 _r1 = _mm_load_ps(p0 + (jj + 1) * batch * 4 + b * 4);
                __m128 _r2 = _mm_load_ps(p0 + (jj + 2) * batch * 4 + b * 4);
                __m128 _r3 = _mm_load_ps(p0 + (jj + 3) * batch * 4 + b * 4);
                __m128 _r4 = _mm_load_ps(p0 + (jj + 4) * batch * 4 + b * 4);
                __m128 _r5 = _mm_load_ps(p0 + (jj + 5) * batch * 4 + b * 4);
                __m128 _r6 = _mm_load_ps(p0 + (jj + 6) * batch * 4 + b * 4);
                __m128 _r7 = _mm_load_ps(p0 + (jj + 7) * batch * 4 + b * 4);
                __m128 _r8 = _mm_load_ps(p0 + (jj + 8) * batch * 4 + b * 4);
                __m128 _r9 = _mm_load_ps(p0 + (jj + 9) * batch * 4 + b * 4);
                __m128 _ra = _mm_load_ps(p0 + (jj + 10) * batch * 4 + b * 4);
                __m128 _rb = _mm_load_ps(p0 + (jj + 11) * batch * 4 + b * 4);
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[jj * batch * 2 + b * 2];
                pp[1] = p0[(jj + 1) * batch * 2 + b * 2];
                pp[2] = p0[(jj + 2) * batch * 2 + b * 2];
                pp[3] = p0[(jj + 3) * batch * 2 + b * 2];
                pp[4] = p0[(jj + 4) * batch * 2 + b * 2];
                pp[5] = p0[(jj + 5) * batch * 2 + b * 2];
                pp[6] = p0[(jj + 6) * batch * 2 + b * 2];
                pp[7] = p0[(jj + 7) * batch * 2 + b * 2];
                pp[8] = p0[(jj + 8) * batch * 2 + b * 2];
                pp[9] = p0[(jj + 9) * batch * 2 + b * 2];
                pp[10] = p0[(jj + 10) * batch * 2 + b * 2];
                pp[11] = p0[(jj + 11) * batch * 2 + b * 2];
                pp[12] = p0[jj * batch * 2 + b * 2 + 1];
                pp[13] = p0[(jj + 1) * batch * 2 + b * 2 + 1];
                pp[14] = p0[(jj + 2) * batch * 2 + b * 2 + 1];
                pp[15] = p0[(jj + 3) * batch * 2 + b * 2 + 1];
                pp[16] = p0[(jj + 4) * batch * 2 + b * 2 + 1];
                pp[17] = p0[(jj + 5) * batch * 2 + b * 2 + 1];
                pp[18] = p0[(jj + 6) * batch * 2 + b * 2 + 1];
                pp[19] = p0[(jj + 7) * batch * 2 + b * 2 + 1];
                pp[20] = p0[(jj + 8) * batch * 2 + b * 2 + 1];
                pp[21] = p0[(jj + 9) * batch * 2 + b * 2 + 1];
                pp[22] = p0[(jj + 10) * batch * 2 + b * 2 + 1];
                pp[23] = p0[(jj + 11) * batch * 2 + b * 2 + 1];
                p0 += max_jj * batch * 2;
                pp += 24;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[jj * batch + b];
                pp[1] = p0[(jj + 1) * batch + b];
                pp[2] = p0[(jj + 2) * batch + b];
                pp[3] = p0[(jj + 3) * batch + b];
                pp[4] = p0[(jj + 4) * batch + b];
                pp[5] = p0[(jj + 5) * batch + b];
                pp[6] = p0[(jj + 6) * batch + b];
                pp[7] = p0[(jj + 7) * batch + b];
                pp[8] = p0[(jj + 8) * batch + b];
                pp[9] = p0[(jj + 9) * batch + b];
                pp[10] = p0[(jj + 10) * batch + b];
                pp[11] = p0[(jj + 11) * batch + b];
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
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0 + jj * batch * 16 + b * 16);
                __m512 _r1 = _mm512_load_ps(p0 + (jj + 1) * batch * 16 + b * 16);
                __m512 _r2 = _mm512_load_ps(p0 + (jj + 2) * batch * 16 + b * 16);
                __m512 _r3 = _mm512_load_ps(p0 + (jj + 3) * batch * 16 + b * 16);
                __m512 _r4 = _mm512_load_ps(p0 + (jj + 4) * batch * 16 + b * 16);
                __m512 _r5 = _mm512_load_ps(p0 + (jj + 5) * batch * 16 + b * 16);
                __m512 _r6 = _mm512_load_ps(p0 + (jj + 6) * batch * 16 + b * 16);
                __m512 _r7 = _mm512_load_ps(p0 + (jj + 7) * batch * 16 + b * 16);
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
#endif // __AVX512F__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0 + jj * batch * 8 + b * 8);
                __m256 _r1 = _mm256_load_ps(p0 + (jj + 1) * batch * 8 + b * 8);
                __m256 _r2 = _mm256_load_ps(p0 + (jj + 2) * batch * 8 + b * 8);
                __m256 _r3 = _mm256_load_ps(p0 + (jj + 3) * batch * 8 + b * 8);
                __m256 _r4 = _mm256_load_ps(p0 + (jj + 4) * batch * 8 + b * 8);
                __m256 _r5 = _mm256_load_ps(p0 + (jj + 5) * batch * 8 + b * 8);
                __m256 _r6 = _mm256_load_ps(p0 + (jj + 6) * batch * 8 + b * 8);
                __m256 _r7 = _mm256_load_ps(p0 + (jj + 7) * batch * 8 + b * 8);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 16, _r2);
                _mm256_storeu_ps(pp + 24, _r3);
                _mm256_storeu_ps(pp + 32, _r4);
                _mm256_storeu_ps(pp + 40, _r5);
                _mm256_storeu_ps(pp + 48, _r6);
                _mm256_storeu_ps(pp + 56, _r7);
                p0 += max_jj * batch * 8;
                pp += 64;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0 + jj * batch * 4 + b * 4);
                __m128 _r1 = _mm_load_ps(p0 + (jj + 1) * batch * 4 + b * 4);
                __m128 _r2 = _mm_load_ps(p0 + (jj + 2) * batch * 4 + b * 4);
                __m128 _r3 = _mm_load_ps(p0 + (jj + 3) * batch * 4 + b * 4);
                __m128 _r4 = _mm_load_ps(p0 + (jj + 4) * batch * 4 + b * 4);
                __m128 _r5 = _mm_load_ps(p0 + (jj + 5) * batch * 4 + b * 4);
                __m128 _r6 = _mm_load_ps(p0 + (jj + 6) * batch * 4 + b * 4);
                __m128 _r7 = _mm_load_ps(p0 + (jj + 7) * batch * 4 + b * 4);
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[jj * batch * 2 + b * 2];
                pp[1] = p0[(jj + 1) * batch * 2 + b * 2];
                pp[2] = p0[(jj + 2) * batch * 2 + b * 2];
                pp[3] = p0[(jj + 3) * batch * 2 + b * 2];
                pp[4] = p0[(jj + 4) * batch * 2 + b * 2];
                pp[5] = p0[(jj + 5) * batch * 2 + b * 2];
                pp[6] = p0[(jj + 6) * batch * 2 + b * 2];
                pp[7] = p0[(jj + 7) * batch * 2 + b * 2];
                pp[8] = p0[jj * batch * 2 + b * 2 + 1];
                pp[9] = p0[(jj + 1) * batch * 2 + b * 2 + 1];
                pp[10] = p0[(jj + 2) * batch * 2 + b * 2 + 1];
                pp[11] = p0[(jj + 3) * batch * 2 + b * 2 + 1];
                pp[12] = p0[(jj + 4) * batch * 2 + b * 2 + 1];
                pp[13] = p0[(jj + 5) * batch * 2 + b * 2 + 1];
                pp[14] = p0[(jj + 6) * batch * 2 + b * 2 + 1];
                pp[15] = p0[(jj + 7) * batch * 2 + b * 2 + 1];
                p0 += max_jj * batch * 2;
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[jj * batch + b];
                pp[1] = p0[(jj + 1) * batch + b];
                pp[2] = p0[(jj + 2) * batch + b];
                pp[3] = p0[(jj + 3) * batch + b];
                pp[4] = p0[(jj + 4) * batch + b];
                pp[5] = p0[(jj + 5) * batch + b];
                pp[6] = p0[(jj + 6) * batch + b];
                pp[7] = p0[(jj + 7) * batch + b];
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
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0 + jj * batch * 16 + b * 16);
                __m512 _r1 = _mm512_load_ps(p0 + (jj + 1) * batch * 16 + b * 16);
                __m512 _r2 = _mm512_load_ps(p0 + (jj + 2) * batch * 16 + b * 16);
                __m512 _r3 = _mm512_load_ps(p0 + (jj + 3) * batch * 16 + b * 16);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm512_storeu_ps(pp, _r0);
                _mm512_storeu_ps(pp + 16, _r1);
                _mm512_storeu_ps(pp + 32, _r2);
                _mm512_storeu_ps(pp + 48, _r3);
                p0 += max_jj * batch * 16;
                pp += 64;
            }
#endif // __AVX512F__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0 + jj * batch * 8 + b * 8);
                __m256 _r1 = _mm256_load_ps(p0 + (jj + 1) * batch * 8 + b * 8);
                __m256 _r2 = _mm256_load_ps(p0 + (jj + 2) * batch * 8 + b * 8);
                __m256 _r3 = _mm256_load_ps(p0 + (jj + 3) * batch * 8 + b * 8);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 16, _r2);
                _mm256_storeu_ps(pp + 24, _r3);
                p0 += max_jj * batch * 8;
                pp += 32;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0 + jj * batch * 4 + b * 4);
                __m128 _r1 = _mm_load_ps(p0 + (jj + 1) * batch * 4 + b * 4);
                __m128 _r2 = _mm_load_ps(p0 + (jj + 2) * batch * 4 + b * 4);
                __m128 _r3 = _mm_load_ps(p0 + (jj + 3) * batch * 4 + b * 4);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4, _r1);
                _mm_store_ps(pp + 8, _r2);
                _mm_store_ps(pp + 12, _r3);
                p0 += max_jj * batch * 4;
                pp += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[jj * batch * 2 + b * 2];
                pp[1] = p0[(jj + 1) * batch * 2 + b * 2];
                pp[2] = p0[(jj + 2) * batch * 2 + b * 2];
                pp[3] = p0[(jj + 3) * batch * 2 + b * 2];
                pp[4] = p0[jj * batch * 2 + b * 2 + 1];
                pp[5] = p0[(jj + 1) * batch * 2 + b * 2 + 1];
                pp[6] = p0[(jj + 2) * batch * 2 + b * 2 + 1];
                pp[7] = p0[(jj + 3) * batch * 2 + b * 2 + 1];
                p0 += max_jj * batch * 2;
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[jj * batch + b];
                pp[1] = p0[(jj + 1) * batch + b];
                pp[2] = p0[(jj + 2) * batch + b];
                pp[3] = p0[(jj + 3) * batch + b];
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
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0 + jj * batch * 16 + b * 16);
                __m512 _r1 = _mm512_load_ps(p0 + (jj + 1) * batch * 16 + b * 16);
                transpose16x2_ps(_r0, _r1);
                _mm512_storeu_ps(pp, _r0);
                _mm512_storeu_ps(pp + 16, _r1);
                p0 += max_jj * batch * 16;
                pp += 32;
            }
#endif // __AVX512F__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0 + jj * batch * 8 + b * 8);
                __m256 _r1 = _mm256_load_ps(p0 + (jj + 1) * batch * 8 + b * 8);
                transpose8x2_ps(_r0, _r1);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                p0 += max_jj * batch * 8;
                pp += 16;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0 + jj * batch * 4 + b * 4);
                __m128 _r1 = _mm_load_ps(p0 + (jj + 1) * batch * 4 + b * 4);
                __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                _mm_store_ps(pp, _tmp0);
                _mm_store_ps(pp + 4, _tmp1);
                p0 += max_jj * batch * 4;
                pp += 8;
            }
#endif // __SSE2__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[jj * batch * 2 + b * 2];
                pp[1] = p0[(jj + 1) * batch * 2 + b * 2];
                pp[2] = p0[jj * batch * 2 + b * 2 + 1];
                pp[3] = p0[(jj + 1) * batch * 2 + b * 2 + 1];
                p0 += max_jj * batch * 2;
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[jj * batch + b];
                pp[1] = p0[(jj + 1) * batch + b];
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
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0 + jj * batch * 16 + b * 16);
                _mm512_storeu_ps(pp, _r0);
                p0 += max_jj * batch * 16;
                pp += 16;
            }
#endif // __AVX512F__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0 + jj * batch * 8 + b * 8);
                _mm256_storeu_ps(pp, _r0);
                p0 += max_jj * batch * 8;
                pp += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0 + jj * batch * 4 + b * 4);
                _mm_storeu_ps(pp, _r0);
                p0 += max_jj * batch * 4;
                pp += 4;
            }
#endif // __SSE2__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[jj * batch * 2 + b * 2];
                pp[1] = p0[jj * batch * 2 + b * 2 + 1];
                p0 += max_jj * batch * 2;
                pp += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[jj * batch + b];
                p0 += max_jj * batch;
                pp += 1;
            }
        }
    }
}

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, Mat& tmp, int batch, int max_ii, int max_jj, int k, int max_kk, bool k_end)
{
    for (int b = 0; b < batch; b++)
    {
        const float* pAT = AT_tile.row(b);
        const float* pBT = BT_tile.row(b);

        Mat outptr = top_blob.depth(b);

        float* ptmp = tmp.row(b);

        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            const float* pB = pBT;

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
                    _sum0 = _mm512_load_ps(ptmp);
                    _sum1 = _mm512_load_ps(ptmp + 16);
                    _sum2 = _mm512_load_ps(ptmp + 16 * 2);
                    _sum3 = _mm512_load_ps(ptmp + 16 * 3);
                    _sum4 = _mm512_load_ps(ptmp + 16 * 4);
                    _sum5 = _mm512_load_ps(ptmp + 16 * 5);
                    _sum6 = _mm512_load_ps(ptmp + 16 * 6);
                    _sum7 = _mm512_load_ps(ptmp + 16 * 7);
                    _sum8 = _mm512_load_ps(ptmp + 16 * 8);
                    _sum9 = _mm512_load_ps(ptmp + 16 * 9);
                    _suma = _mm512_load_ps(ptmp + 16 * 10);
                    _sumb = _mm512_load_ps(ptmp + 16 * 11);
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

                if (k_end)
                {
                    _mm512_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm512_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm512_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm512_store_ps(outptr.row(jj + 3) + ii, _sum3);
                    _mm512_store_ps(outptr.row(jj + 4) + ii, _sum4);
                    _mm512_store_ps(outptr.row(jj + 5) + ii, _sum5);
                    _mm512_store_ps(outptr.row(jj + 6) + ii, _sum6);
                    _mm512_store_ps(outptr.row(jj + 7) + ii, _sum7);
                    _mm512_store_ps(outptr.row(jj + 8) + ii, _sum8);
                    _mm512_store_ps(outptr.row(jj + 9) + ii, _sum9);
                    _mm512_store_ps(outptr.row(jj + 10) + ii, _suma);
                    _mm512_store_ps(outptr.row(jj + 11) + ii, _sumb);
                }
                else
                {
                    _mm512_store_ps(ptmp, _sum0);
                    _mm512_store_ps(ptmp + 16, _sum1);
                    _mm512_store_ps(ptmp + 16 * 2, _sum2);
                    _mm512_store_ps(ptmp + 16 * 3, _sum3);
                    _mm512_store_ps(ptmp + 16 * 4, _sum4);
                    _mm512_store_ps(ptmp + 16 * 5, _sum5);
                    _mm512_store_ps(ptmp + 16 * 6, _sum6);
                    _mm512_store_ps(ptmp + 16 * 7, _sum7);
                    _mm512_store_ps(ptmp + 16 * 8, _sum8);
                    _mm512_store_ps(ptmp + 16 * 9, _sum9);
                    _mm512_store_ps(ptmp + 16 * 10, _suma);
                    _mm512_store_ps(ptmp + 16 * 11, _sumb);
                }

                ptmp += 192;
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
                    _sum0 = _mm512_load_ps(ptmp);
                    _sum1 = _mm512_load_ps(ptmp + 16);
                    _sum2 = _mm512_load_ps(ptmp + 16 * 2);
                    _sum3 = _mm512_load_ps(ptmp + 16 * 3);
                    _sum4 = _mm512_load_ps(ptmp + 16 * 4);
                    _sum5 = _mm512_load_ps(ptmp + 16 * 5);
                    _sum6 = _mm512_load_ps(ptmp + 16 * 6);
                    _sum7 = _mm512_load_ps(ptmp + 16 * 7);
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

                if (k_end)
                {
                    _mm512_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm512_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm512_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm512_store_ps(outptr.row(jj + 3) + ii, _sum3);
                    _mm512_store_ps(outptr.row(jj + 4) + ii, _sum4);
                    _mm512_store_ps(outptr.row(jj + 5) + ii, _sum5);
                    _mm512_store_ps(outptr.row(jj + 6) + ii, _sum6);
                    _mm512_store_ps(outptr.row(jj + 7) + ii, _sum7);
                }
                else
                {
                    _mm512_store_ps(ptmp, _sum0);
                    _mm512_store_ps(ptmp + 16, _sum1);
                    _mm512_store_ps(ptmp + 16 * 2, _sum2);
                    _mm512_store_ps(ptmp + 16 * 3, _sum3);
                    _mm512_store_ps(ptmp + 16 * 4, _sum4);
                    _mm512_store_ps(ptmp + 16 * 5, _sum5);
                    _mm512_store_ps(ptmp + 16 * 6, _sum6);
                    _mm512_store_ps(ptmp + 16 * 7, _sum7);
                }

                ptmp += 128;
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
                    _sum0 = _mm512_load_ps(ptmp);
                    _sum1 = _mm512_load_ps(ptmp + 16);
                    _sum2 = _mm512_load_ps(ptmp + 16 * 2);
                    _sum3 = _mm512_load_ps(ptmp + 16 * 3);
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

                if (k_end)
                {
                    _mm512_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm512_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm512_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm512_store_ps(outptr.row(jj + 3) + ii, _sum3);
                }
                else
                {
                    _mm512_store_ps(ptmp, _sum0);
                    _mm512_store_ps(ptmp + 16, _sum1);
                    _mm512_store_ps(ptmp + 16 * 2, _sum2);
                    _mm512_store_ps(ptmp + 16 * 3, _sum3);
                }

                ptmp += 64;
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
                    _sum0 = _mm512_load_ps(ptmp);
                    _sum1 = _mm512_load_ps(ptmp + 8);
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

                if (k_end)
                {
                    _mm512_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm512_store_ps(outptr.row(jj + 1) + ii, _sum1);
                }
                else
                {
                    _mm512_store_ps(ptmp, _sum0);
                    _mm512_store_ps(ptmp + 16, _sum1);
                }

                ptmp += 32;
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
                    _sum = _mm512_load_ps(ptmp);
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

                if (k_end)
                {
                    _mm512_store_ps(outptr.row(jj) + ii, _sum);
                }
                else
                {
                    _mm512_store_ps(ptmp, _sum);
                }

                ptmp += 16;
            }

            pAT += max_kk * 16;
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* pB = pBT;

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
                    _sum0 = _mm256_load_ps(ptmp);
                    _sum1 = _mm256_load_ps(ptmp + 8);
                    _sum2 = _mm256_load_ps(ptmp + 16);
                    _sum3 = _mm256_load_ps(ptmp + 24);
                    _sum4 = _mm256_load_ps(ptmp + 32);
                    _sum5 = _mm256_load_ps(ptmp + 40);
                    _sum6 = _mm256_load_ps(ptmp + 48);
                    _sum7 = _mm256_load_ps(ptmp + 56);
                    _sum8 = _mm256_load_ps(ptmp + 64);
                    _sum9 = _mm256_load_ps(ptmp + 72);
                    _suma = _mm256_load_ps(ptmp + 80);
                    _sumb = _mm256_load_ps(ptmp + 88);
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

                if (k_end)
                {
                    _mm256_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm256_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm256_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm256_store_ps(outptr.row(jj + 3) + ii, _sum3);
                    _mm256_store_ps(outptr.row(jj + 4) + ii, _sum4);
                    _mm256_store_ps(outptr.row(jj + 5) + ii, _sum5);
                    _mm256_store_ps(outptr.row(jj + 6) + ii, _sum6);
                    _mm256_store_ps(outptr.row(jj + 7) + ii, _sum7);
                    _mm256_store_ps(outptr.row(jj + 8) + ii, _sum8);
                    _mm256_store_ps(outptr.row(jj + 9) + ii, _sum9);
                    _mm256_store_ps(outptr.row(jj + 10) + ii, _suma);
                    _mm256_store_ps(outptr.row(jj + 11) + ii, _sumb);
                }
                else
                {
                    _mm256_store_ps(ptmp, _sum0);
                    _mm256_store_ps(ptmp + 8, _sum1);
                    _mm256_store_ps(ptmp + 16, _sum2);
                    _mm256_store_ps(ptmp + 24, _sum3);
                    _mm256_store_ps(ptmp + 32, _sum4);
                    _mm256_store_ps(ptmp + 40, _sum5);
                    _mm256_store_ps(ptmp + 48, _sum6);
                    _mm256_store_ps(ptmp + 56, _sum7);
                    _mm256_store_ps(ptmp + 64, _sum8);
                    _mm256_store_ps(ptmp + 72, _sum9);
                    _mm256_store_ps(ptmp + 80, _suma);
                    _mm256_store_ps(ptmp + 88, _sumb);
                }

                ptmp += 96;
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
                    _sum0 = _mm256_load_ps(ptmp);
                    _sum1 = _mm256_load_ps(ptmp + 8);
                    _sum2 = _mm256_load_ps(ptmp + 16);
                    _sum3 = _mm256_load_ps(ptmp + 24);
                    _sum4 = _mm256_load_ps(ptmp + 32);
                    _sum5 = _mm256_load_ps(ptmp + 40);
                    _sum6 = _mm256_load_ps(ptmp + 48);
                    _sum7 = _mm256_load_ps(ptmp + 56);
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

                if (k_end)
                {
                    _mm256_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm256_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm256_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm256_store_ps(outptr.row(jj + 3) + ii, _sum3);
                    _mm256_store_ps(outptr.row(jj + 4) + ii, _sum4);
                    _mm256_store_ps(outptr.row(jj + 5) + ii, _sum5);
                    _mm256_store_ps(outptr.row(jj + 6) + ii, _sum6);
                    _mm256_store_ps(outptr.row(jj + 7) + ii, _sum7);
                }
                else
                {
                    _mm256_store_ps(ptmp, _sum0);
                    _mm256_store_ps(ptmp + 8, _sum1);
                    _mm256_store_ps(ptmp + 16, _sum2);
                    _mm256_store_ps(ptmp + 24, _sum3);
                    _mm256_store_ps(ptmp + 32, _sum4);
                    _mm256_store_ps(ptmp + 40, _sum5);
                    _mm256_store_ps(ptmp + 48, _sum6);
                    _mm256_store_ps(ptmp + 56, _sum7);
                }

                ptmp += 64;
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
                    _sum0 = _mm256_load_ps(ptmp);
                    _sum1 = _mm256_load_ps(ptmp + 8);
                    _sum2 = _mm256_load_ps(ptmp + 16);
                    _sum3 = _mm256_load_ps(ptmp + 24);
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

                if (k_end)
                {
                    _mm256_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm256_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm256_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm256_store_ps(outptr.row(jj + 3) + ii, _sum3);
                }
                else
                {
                    _mm256_store_ps(ptmp, _sum0);
                    _mm256_store_ps(ptmp + 8, _sum1);
                    _mm256_store_ps(ptmp + 16, _sum2);
                    _mm256_store_ps(ptmp + 24, _sum3);
                }

                ptmp += 32;
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
                    _sum0 = _mm256_load_ps(ptmp);
                    _sum1 = _mm256_load_ps(ptmp + 8);
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

                if (k_end)
                {
                    _mm256_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm256_store_ps(outptr.row(jj + 1) + ii, _sum1);
                }
                else
                {
                    _mm256_store_ps(ptmp, _sum0);
                    _mm256_store_ps(ptmp + 8, _sum1);
                }

                ptmp += 16;
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
                    _sum = _mm256_load_ps(ptmp);
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

                if (k_end)
                {
                    _mm256_store_ps(outptr.row(jj) + ii, _sum);
                }
                else
                {
                    _mm256_store_ps(ptmp, _sum);
                }

                ptmp += 8;
            }

            pAT += max_kk * 8;
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* pB = pBT;

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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
                    _sum2 = _mm_load_ps(ptmp + 8);
                    _sum3 = _mm_load_ps(ptmp + 12);
                    _sum4 = _mm_load_ps(ptmp + 16);
                    _sum5 = _mm_load_ps(ptmp + 20);
                    _sum6 = _mm_load_ps(ptmp + 24);
                    _sum7 = _mm_load_ps(ptmp + 28);
                    _sum8 = _mm_load_ps(ptmp + 32);
                    _sum9 = _mm_load_ps(ptmp + 36);
                    _suma = _mm_load_ps(ptmp + 40);
                    _sumb = _mm_load_ps(ptmp + 44);
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

                if (k_end)
                {
                    _mm_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm_store_ps(outptr.row(jj + 3) + ii, _sum3);
                    _mm_store_ps(outptr.row(jj + 4) + ii, _sum4);
                    _mm_store_ps(outptr.row(jj + 5) + ii, _sum5);
                    _mm_store_ps(outptr.row(jj + 6) + ii, _sum6);
                    _mm_store_ps(outptr.row(jj + 7) + ii, _sum7);
                    _mm_store_ps(outptr.row(jj + 8) + ii, _sum8);
                    _mm_store_ps(outptr.row(jj + 9) + ii, _sum9);
                    _mm_store_ps(outptr.row(jj + 10) + ii, _suma);
                    _mm_store_ps(outptr.row(jj + 11) + ii, _sumb);
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                    _mm_store_ps(ptmp + 8, _sum2);
                    _mm_store_ps(ptmp + 12, _sum3);
                    _mm_store_ps(ptmp + 16, _sum4);
                    _mm_store_ps(ptmp + 20, _sum5);
                    _mm_store_ps(ptmp + 24, _sum6);
                    _mm_store_ps(ptmp + 28, _sum7);
                    _mm_store_ps(ptmp + 32, _sum8);
                    _mm_store_ps(ptmp + 36, _sum9);
                    _mm_store_ps(ptmp + 40, _suma);
                    _mm_store_ps(ptmp + 44, _sumb);
                }

                ptmp += 48;
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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
                    _sum2 = _mm_load_ps(ptmp + 8);
                    _sum3 = _mm_load_ps(ptmp + 12);
                    _sum4 = _mm_load_ps(ptmp + 16);
                    _sum5 = _mm_load_ps(ptmp + 20);
                    _sum6 = _mm_load_ps(ptmp + 24);
                    _sum7 = _mm_load_ps(ptmp + 28);
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

                if (k_end)
                {
                    _mm_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm_store_ps(outptr.row(jj + 3) + ii, _sum3);
                    _mm_store_ps(outptr.row(jj + 4) + ii, _sum4);
                    _mm_store_ps(outptr.row(jj + 5) + ii, _sum5);
                    _mm_store_ps(outptr.row(jj + 6) + ii, _sum6);
                    _mm_store_ps(outptr.row(jj + 7) + ii, _sum7);
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                    _mm_store_ps(ptmp + 8, _sum2);
                    _mm_store_ps(ptmp + 12, _sum3);
                    _mm_store_ps(ptmp + 16, _sum4);
                    _mm_store_ps(ptmp + 20, _sum5);
                    _mm_store_ps(ptmp + 24, _sum6);
                    _mm_store_ps(ptmp + 28, _sum7);
                }

                ptmp += 32;
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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
                    _sum2 = _mm_load_ps(ptmp + 8);
                    _sum3 = _mm_load_ps(ptmp + 12);
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

                if (k_end)
                {
                    _mm_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm_store_ps(outptr.row(jj + 1) + ii, _sum1);
                    _mm_store_ps(outptr.row(jj + 2) + ii, _sum2);
                    _mm_store_ps(outptr.row(jj + 3) + ii, _sum3);
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                    _mm_store_ps(ptmp + 8, _sum2);
                    _mm_store_ps(ptmp + 12, _sum3);
                }

                ptmp += 16;
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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
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

                if (k_end)
                {
                    _mm_store_ps(outptr.row(jj) + ii, _sum0);
                    _mm_store_ps(outptr.row(jj + 1) + ii, _sum1);
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                }

                ptmp += 8;
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
                    _sum = _mm_load_ps(ptmp);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = _mm_load_ps(pA);
                    _sum = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum);

                    pA += 4;
                    pB += 1;
                }

                if (k_end)
                {
                    _mm_store_ps(outptr.row(jj) + ii, _sum);
                }
                else
                {
                    _mm_store_ps(ptmp, _sum);
                }

                ptmp += 4;
            }

            pAT += max_kk * 4;
        }
#endif // __SSE2__
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* pB = pBT;

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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
                    _sum2 = _mm_load_ps(ptmp + 8);
                    _sum3 = _mm_load_ps(ptmp + 12);
                    _sum4 = _mm_load_ps(ptmp + 16);
                    _sum5 = _mm_load_ps(ptmp + 20);
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

                if (k_end)
                {
                    float sum[24];
                    _mm_storeu_ps(sum, _sum0);
                    _mm_storeu_ps(sum + 4, _sum1);
                    _mm_storeu_ps(sum + 8, _sum2);
                    _mm_storeu_ps(sum + 12, _sum3);
                    _mm_storeu_ps(sum + 16, _sum4);
                    _mm_storeu_ps(sum + 20, _sum5);
                    outptr.row(jj)[ii] = sum[0];
                    outptr.row(jj + 1)[ii] = sum[1];
                    outptr.row(jj + 2)[ii] = sum[2];
                    outptr.row(jj + 3)[ii] = sum[3];
                    outptr.row(jj + 4)[ii] = sum[4];
                    outptr.row(jj + 5)[ii] = sum[5];
                    outptr.row(jj + 6)[ii] = sum[6];
                    outptr.row(jj + 7)[ii] = sum[7];
                    outptr.row(jj + 8)[ii] = sum[8];
                    outptr.row(jj + 9)[ii] = sum[9];
                    outptr.row(jj + 10)[ii] = sum[10];
                    outptr.row(jj + 11)[ii] = sum[11];
                    outptr.row(jj)[ii + 1] = sum[12];
                    outptr.row(jj + 1)[ii + 1] = sum[13];
                    outptr.row(jj + 2)[ii + 1] = sum[14];
                    outptr.row(jj + 3)[ii + 1] = sum[15];
                    outptr.row(jj + 4)[ii + 1] = sum[16];
                    outptr.row(jj + 5)[ii + 1] = sum[17];
                    outptr.row(jj + 6)[ii + 1] = sum[18];
                    outptr.row(jj + 7)[ii + 1] = sum[19];
                    outptr.row(jj + 8)[ii + 1] = sum[20];
                    outptr.row(jj + 9)[ii + 1] = sum[21];
                    outptr.row(jj + 10)[ii + 1] = sum[22];
                    outptr.row(jj + 11)[ii + 1] = sum[23];
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                    _mm_store_ps(ptmp + 8, _sum2);
                    _mm_store_ps(ptmp + 12, _sum3);
                    _mm_store_ps(ptmp + 16, _sum4);
                    _mm_store_ps(ptmp + 20, _sum5);
                }

                ptmp += 24;
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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
                    _sum2 = _mm_load_ps(ptmp + 8);
                    _sum3 = _mm_load_ps(ptmp + 12);
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

                if (k_end)
                {
                    float sum[16];
                    _mm_storeu_ps(sum, _sum0);
                    _mm_storeu_ps(sum + 4, _sum1);
                    _mm_storeu_ps(sum + 8, _sum2);
                    _mm_storeu_ps(sum + 12, _sum3);
                    outptr.row(jj)[ii] = sum[0];
                    outptr.row(jj + 1)[ii] = sum[1];
                    outptr.row(jj + 2)[ii] = sum[2];
                    outptr.row(jj + 3)[ii] = sum[3];
                    outptr.row(jj + 4)[ii] = sum[4];
                    outptr.row(jj + 5)[ii] = sum[5];
                    outptr.row(jj + 6)[ii] = sum[6];
                    outptr.row(jj + 7)[ii] = sum[7];
                    outptr.row(jj)[ii + 1] = sum[8];
                    outptr.row(jj + 1)[ii + 1] = sum[9];
                    outptr.row(jj + 2)[ii + 1] = sum[10];
                    outptr.row(jj + 3)[ii + 1] = sum[11];
                    outptr.row(jj + 4)[ii + 1] = sum[12];
                    outptr.row(jj + 5)[ii + 1] = sum[13];
                    outptr.row(jj + 6)[ii + 1] = sum[14];
                    outptr.row(jj + 7)[ii + 1] = sum[15];
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                    _mm_store_ps(ptmp + 8, _sum2);
                    _mm_store_ps(ptmp + 12, _sum3);
                }

                ptmp += 16;
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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
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

                if (k_end)
                {
                    float sum[8];
                    _mm_storeu_ps(sum, _sum0);
                    _mm_storeu_ps(sum + 4, _sum1);
                    outptr.row(jj)[ii] = sum[0];
                    outptr.row(jj + 1)[ii] = sum[1];
                    outptr.row(jj + 2)[ii] = sum[2];
                    outptr.row(jj + 3)[ii] = sum[3];
                    outptr.row(jj)[ii + 1] = sum[4];
                    outptr.row(jj + 1)[ii + 1] = sum[5];
                    outptr.row(jj + 2)[ii + 1] = sum[6];
                    outptr.row(jj + 3)[ii + 1] = sum[7];
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                }

                ptmp += 8;
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
                    sum00 = ptmp[0];
                    sum01 = ptmp[1];
                    sum10 = ptmp[2];
                    sum11 = ptmp[3];
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

                if (k_end)
                {
                    outptr.row(jj)[ii] = sum00;
                    outptr.row(jj)[ii + 1] = sum01;
                    outptr.row(jj + 1)[ii] = sum10;
                    outptr.row(jj + 1)[ii + 1] = sum11;
                }
                else
                {
                    ptmp[0] = sum00;
                    ptmp[1] = sum01;
                    ptmp[2] = sum10;
                    ptmp[3] = sum11;
                }

                ptmp += 4;
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
                    sum0 = ptmp[0];
                    sum1 = ptmp[1];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[1] * pB[0];
                    pA += 2;
                    pB += 1;
                }

                if (k_end)
                {
                    outptr.row(jj)[ii] = sum0;
                    outptr.row(jj)[ii + 1] = sum1;
                }
                else
                {
                    ptmp[0] = sum0;
                    ptmp[1] = sum1;
                }

                ptmp += 2;
            }

            pAT += max_kk * 2;
        }
        for (; ii < max_ii; ii++)
        {
            const float* pB = pBT;

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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
                    _sum2 = _mm_load_ps(ptmp + 8);
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

                if (k_end)
                {
                    float sum[12];
                    _mm_storeu_ps(sum, _sum0);
                    _mm_storeu_ps(sum + 4, _sum1);
                    _mm_storeu_ps(sum + 8, _sum2);
                    outptr.row(jj)[ii] = sum[0];
                    outptr.row(jj + 1)[ii] = sum[1];
                    outptr.row(jj + 2)[ii] = sum[2];
                    outptr.row(jj + 3)[ii] = sum[3];
                    outptr.row(jj + 4)[ii] = sum[4];
                    outptr.row(jj + 5)[ii] = sum[5];
                    outptr.row(jj + 6)[ii] = sum[6];
                    outptr.row(jj + 7)[ii] = sum[7];
                    outptr.row(jj + 8)[ii] = sum[8];
                    outptr.row(jj + 9)[ii] = sum[9];
                    outptr.row(jj + 10)[ii] = sum[10];
                    outptr.row(jj + 11)[ii] = sum[11];
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                    _mm_store_ps(ptmp + 8, _sum2);
                }

                ptmp += 12;
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
                    _sum0 = _mm_load_ps(ptmp);
                    _sum1 = _mm_load_ps(ptmp + 4);
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

                if (k_end)
                {
                    float sum[8];
                    _mm_storeu_ps(sum, _sum0);
                    _mm_storeu_ps(sum + 4, _sum1);
                    outptr.row(jj)[ii] = sum[0];
                    outptr.row(jj + 1)[ii] = sum[1];
                    outptr.row(jj + 2)[ii] = sum[2];
                    outptr.row(jj + 3)[ii] = sum[3];
                    outptr.row(jj + 4)[ii] = sum[4];
                    outptr.row(jj + 5)[ii] = sum[5];
                    outptr.row(jj + 6)[ii] = sum[6];
                    outptr.row(jj + 7)[ii] = sum[7];
                }
                else
                {
                    _mm_store_ps(ptmp, _sum0);
                    _mm_store_ps(ptmp + 4, _sum1);
                }

                ptmp += 8;
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
                    _sum = _mm_load_ps(ptmp);
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

                if (k_end)
                {
                    float sum[4];
                    _mm_storeu_ps(sum, _sum);
                    outptr.row(jj)[ii] = sum[0];
                    outptr.row(jj + 1)[ii] = sum[1];
                    outptr.row(jj + 2)[ii] = sum[2];
                    outptr.row(jj + 3)[ii] = sum[3];
                }
                else
                {
                    _mm_store_ps(ptmp, _sum);
                }

                ptmp += 4;
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
                    sum0 = ptmp[0];
                    sum1 = ptmp[1];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[0] * pB[1];
                    pA += 1;
                    pB += 2;
                }

                if (k_end)
                {
                    outptr.row(jj)[ii] = sum0;
                    outptr.row(jj + 1)[ii] = sum1;
                }
                else
                {
                    ptmp[0] = sum0;
                    ptmp[1] = sum1;
                }

                ptmp += 2;
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
                    sum = ptmp[0];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum += pA[0] * pB[0];
                    pA += 1;
                    pB += 1;
                }

                if (k_end)
                {
                    outptr.row(jj)[ii] = sum;
                }
                else
                {
                    ptmp[0] = sum;
                }

                ptmp += 1;
            }

            pAT += max_kk;
        }
    }
}

static void get_optimal_tile_mnk(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    size_t l2_cache_size = get_cpu_level2_cache_size();
    int tile_size = (int)sqrt((float)l2_cache_size / 3 / sizeof(float));

#if __AVX512F__
    TILE_M = tile_size / 16 * 16;
    TILE_N = tile_size / 4 * 4;
    TILE_K = tile_size / 16 * 16;
#elif __AVX__
    TILE_M = tile_size / 8 * 8;
    TILE_N = tile_size / 4 * 4;
    TILE_K = tile_size / 8 * 8;
#elif __SSE2__
    TILE_M = tile_size / 4 * 4;
    TILE_N = tile_size / 4 * 4;
    TILE_K = tile_size / 4 * 4;
#else
    TILE_M = tile_size / 2 * 2;
    TILE_N = tile_size;
    TILE_K = tile_size / 2 * 2;
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
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(float) / TILE_K);

#if __AVX512F__
            TILE_M = tile_size / 16 * 16;
            TILE_N = tile_size / 4 * 4;
#elif __AVX__
            TILE_M = tile_size / 8 * 8;
            TILE_N = tile_size / 4 * 4;
#elif __SSE2__
            TILE_M = tile_size / 4 * 4;
            TILE_N = tile_size / 4 * 4;
#else
            TILE_M = tile_size / 2 * 2;
            TILE_N = tile_size;
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
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __AVX__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
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

            for (int m = 0; m < 3; m++)
            {
                const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9 + m * 3;

                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = r0 * 0.5f + r1 * 0.5f + r2 * 0.5f;
                tmp[2][m] = r0 * 0.5f - r1 * 0.5f + r2 * 0.5f;
                tmp[3][m] = r2;
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

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, 4u, opt.blob_allocator);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 4u, opt.blob_allocator);

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

static inline void conv3x3s1_winograd23_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
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

    const int w_tiles = (w - 1) / 2;

    float* ptmp = B;

    int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; kk + 15 < max_kk; kk += 16)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(64))
#else
            __attribute__((aligned(64)))
#endif
            float tmp[4][4][16];

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
                        const float* r0 = bottom_blob.channel((k + kk) / 16).row(ti * 2 + m) + (tj * 2) * 16;

                        _r0 = _mm512_load_ps(r0);
                        if (tj * 2 + 1 < w) _r1 = _mm512_load_ps(r0 + 16);
                        if (tj * 2 + 2 < w) _r2 = _mm512_load_ps(r0 + 32);
                        if (tj * 2 + 3 < w) _r3 = _mm512_load_ps(r0 + 48);
                    }
                    if (elempack == 8)
                    {
                        const float* r0 = bottom_blob.channel((k + kk) / 8).row(ti * 2 + m) + (tj * 2) * 8;
                        const float* r1 = bottom_blob.channel((k + kk) / 8 + 1).row(ti * 2 + m) + (tj * 2) * 8;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0)), _mm256_load_ps(r1), 1);
                        if (tj * 2 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 8)), _mm256_load_ps(r1 + 8), 1);
                        if (tj * 2 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 16)), _mm256_load_ps(r1 + 16), 1);
                        if (tj * 2 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 24)), _mm256_load_ps(r1 + 24), 1);
                    }
                    if (elempack == 4)
                    {
                        const float* r0 = bottom_blob.channel((k + kk) / 4).row(ti * 2 + m) + (tj * 2) * 4;
                        const float* r1 = bottom_blob.channel((k + kk) / 4 + 1).row(ti * 2 + m) + (tj * 2) * 4;
                        const float* r2 = bottom_blob.channel((k + kk) / 4 + 2).row(ti * 2 + m) + (tj * 2) * 4;
                        const float* r3 = bottom_blob.channel((k + kk) / 4 + 3).row(ti * 2 + m) + (tj * 2) * 4;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2)), _mm_load_ps(r3), 1), 1);
                        if (tj * 2 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 4)), _mm_load_ps(r3 + 4), 1), 1);
                        if (tj * 2 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 8)), _mm_load_ps(r3 + 8), 1), 1);
                        if (tj * 2 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 12)), _mm_load_ps(r3 + 12), 1), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r0 = bottom_blob.channel(k + kk).row(ti * 2 + m) + (tj * 2);
                        const float* r1 = bottom_blob.channel(k + kk + 1).row(ti * 2 + m) + (tj * 2);
                        const float* r2 = bottom_blob.channel(k + kk + 2).row(ti * 2 + m) + (tj * 2);
                        const float* r3 = bottom_blob.channel(k + kk + 3).row(ti * 2 + m) + (tj * 2);
                        const float* r4 = bottom_blob.channel(k + kk + 4).row(ti * 2 + m) + (tj * 2);
                        const float* r5 = bottom_blob.channel(k + kk + 5).row(ti * 2 + m) + (tj * 2);
                        const float* r6 = bottom_blob.channel(k + kk + 6).row(ti * 2 + m) + (tj * 2);
                        const float* r7 = bottom_blob.channel(k + kk + 7).row(ti * 2 + m) + (tj * 2);
                        const float* r8 = bottom_blob.channel(k + kk + 8).row(ti * 2 + m) + (tj * 2);
                        const float* r9 = bottom_blob.channel(k + kk + 9).row(ti * 2 + m) + (tj * 2);
                        const float* ra = bottom_blob.channel(k + kk + 10).row(ti * 2 + m) + (tj * 2);
                        const float* rb = bottom_blob.channel(k + kk + 11).row(ti * 2 + m) + (tj * 2);
                        const float* rc = bottom_blob.channel(k + kk + 12).row(ti * 2 + m) + (tj * 2);
                        const float* rd = bottom_blob.channel(k + kk + 13).row(ti * 2 + m) + (tj * 2);
                        const float* re = bottom_blob.channel(k + kk + 14).row(ti * 2 + m) + (tj * 2);
                        const float* rf = bottom_blob.channel(k + kk + 15).row(ti * 2 + m) + (tj * 2);

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
            }
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

                _mm512_store_ps(ptmp, _tmp0);
                _mm512_store_ps(ptmp + 16, _tmp1);
                _mm512_store_ps(ptmp + 32, _tmp2);
                _mm512_store_ps(ptmp + 48, _tmp3);
                ptmp += 64;
            }
        }
    }
#endif // __AVX512F__
    for (; kk + 7 < max_kk; kk += 8)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[4][4][8];

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
                        const float* r0 = bottom_blob.channel((k + kk) / 8).row(ti * 2 + m) + (tj * 2) * 8;

                        _r0 = _mm256_load_ps(r0);
                        if (tj * 2 + 1 < w) _r1 = _mm256_load_ps(r0 + 8);
                        if (tj * 2 + 2 < w) _r2 = _mm256_load_ps(r0 + 16);
                        if (tj * 2 + 3 < w) _r3 = _mm256_load_ps(r0 + 24);
                    }
                    if (elempack == 4)
                    {
                        const float* r0 = bottom_blob.channel((k + kk) / 4).row(ti * 2 + m) + (tj * 2) * 4;
                        const float* r1 = bottom_blob.channel((k + kk) / 4 + 1).row(ti * 2 + m) + (tj * 2) * 4;

                        _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1);
                        if (tj * 2 + 1 < w) _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1);
                        if (tj * 2 + 2 < w) _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1);
                        if (tj * 2 + 3 < w) _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r0 = bottom_blob.channel(k + kk).row(ti * 2 + m) + (tj * 2);
                        const float* r1 = bottom_blob.channel(k + kk + 1).row(ti * 2 + m) + (tj * 2);
                        const float* r2 = bottom_blob.channel(k + kk + 2).row(ti * 2 + m) + (tj * 2);
                        const float* r3 = bottom_blob.channel(k + kk + 3).row(ti * 2 + m) + (tj * 2);
                        const float* r4 = bottom_blob.channel(k + kk + 4).row(ti * 2 + m) + (tj * 2);
                        const float* r5 = bottom_blob.channel(k + kk + 5).row(ti * 2 + m) + (tj * 2);
                        const float* r6 = bottom_blob.channel(k + kk + 6).row(ti * 2 + m) + (tj * 2);
                        const float* r7 = bottom_blob.channel(k + kk + 7).row(ti * 2 + m) + (tj * 2);

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
            }
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

                _mm256_store_ps(ptmp, _tmp0);
                _mm256_store_ps(ptmp + 8, _tmp1);
                _mm256_store_ps(ptmp + 16, _tmp2);
                _mm256_store_ps(ptmp + 24, _tmp3);
                ptmp += 32;
            }
        }
    }
#endif // __AVX__
    for (; kk + 3 < max_kk; kk += 4)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(16))
#else
            __attribute__((aligned(16)))
#endif
            float tmp[4][4][4];

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
                        const float* r0 = bottom_blob.channel((k + kk) / 4).row(ti * 2 + m) + (tj * 2) * 4;

                        _r0 = _mm_load_ps(r0);
                        if (tj * 2 + 1 < w) _r1 = _mm_load_ps(r0 + 4);
                        if (tj * 2 + 2 < w) _r2 = _mm_load_ps(r0 + 8);
                        if (tj * 2 + 3 < w) _r3 = _mm_load_ps(r0 + 12);
                    }
                    if (elempack == 1)
                    {
                        const float* r0 = bottom_blob.channel(k + kk).row(ti * 2 + m) + (tj * 2);
                        const float* r1 = bottom_blob.channel(k + kk + 1).row(ti * 2 + m) + (tj * 2);
                        const float* r2 = bottom_blob.channel(k + kk + 2).row(ti * 2 + m) + (tj * 2);
                        const float* r3 = bottom_blob.channel(k + kk + 3).row(ti * 2 + m) + (tj * 2);

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
            }
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

                _mm_store_ps(ptmp, _tmp0);
                _mm_store_ps(ptmp + 4, _tmp1);
                _mm_store_ps(ptmp + 8, _tmp2);
                _mm_store_ps(ptmp + 12, _tmp3);
                ptmp += 16;
            }
        }
    }
#endif // __SSE2__
    for (; kk + 1 < max_kk; kk += 2)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            float tmp[4][4][2];

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
                        const float* r0123_0 = bottom_blob.channel(k + kk).row(ti * 2 + m) + (tj * 2);
                        const float* r0123_1 = bottom_blob.channel(k + kk + 1).row(ti * 2 + m) + (tj * 2);

                        r00 = r0123_0[0];
                        r01 = r0123_1[0];
                        if (tj * 2 + 1 < w)
                        {
                            r10 = r0123_0[1];
                            r11 = r0123_1[1];
                        }
                        if (tj * 2 + 2 < w)
                        {
                            r20 = r0123_0[2];
                            r21 = r0123_1[2];
                        }
                        if (tj * 2 + 3 < w)
                        {
                            r30 = r0123_0[3];
                            r31 = r0123_1[3];
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
            }
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

                float z00 = r00 - r20;
                float z01 = r01 - r21;
                float z10 = r10 + r20;
                float z11 = r11 + r21;
                float z20 = r20 - r10;
                float z21 = r21 - r11;
                float z30 = r30 - r10;
                float z31 = r31 - r11;

                ptmp[0] = z00;
                ptmp[1] = z01;
                ptmp[2] = z10;
                ptmp[3] = z11;
                ptmp[4] = z20;
                ptmp[5] = z21;
                ptmp[6] = z30;
                ptmp[7] = z31;
                ptmp += 8;
            }
        }
    }
    for (; kk < max_kk; kk++)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            float tmp[4][4];

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
                        const float* r0123 = bottom_blob.channel(k + kk).row(ti * 2 + m) + (tj * 2);

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
            }
            for (int m = 0; m < 4; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

                float z0 = r0 - r2;
                float z1 = r1 + r2;
                float z2 = r2 - r1;
                float z3 = r3 - r1;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp += 4;
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

    const int w_tiles = (outw + 1) / 2;

    const float* biasptr = bias;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        __m512 _bias0 = biasptr ? _mm512_loadu_ps(biasptr + i + ii) : _mm512_setzero_ps();

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(64))
#else
            __attribute__((aligned(64)))
#endif
            float tmp[2][4][16];

            for (int m = 0; m < 4; m++)
            {
                const float* r0 = top_tile.depth(m * 4).row(jj) + ii;
                const float* r1 = top_tile.depth(m * 4 + 1).row(jj) + ii;
                const float* r2 = top_tile.depth(m * 4 + 2).row(jj) + ii;
                const float* r3 = top_tile.depth(m * 4 + 3).row(jj) + ii;

                __m512 _r0 = _mm512_load_ps(r0);
                __m512 _r1 = _mm512_load_ps(r1);
                __m512 _r2 = _mm512_load_ps(r2);
                __m512 _r3 = _mm512_load_ps(r3);

                __m512 _tmp0 = _mm512_add_ps(_mm512_add_ps(_r0, _r1), _r2);
                __m512 _tmp1 = _mm512_add_ps(_mm512_sub_ps(_r1, _r2), _r3);

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);
            }
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
                    float* output0 = top_blob.channel((i + ii) / 16).row(ti * 2 + m) + (tj * 2) * 16;

                    _mm512_store_ps(output0, _tmp0);
                    if (tj * 2 + 1 < outw)
                    {
                        _mm512_store_ps(output0 + 16, _tmp1);
                    }
                }
                if (out_elempack == 8)
                {
                    float* output0 = top_blob.channel((i + ii) / 8).row(ti * 2 + m) + (tj * 2) * 8;
                    float* output1 = top_blob.channel((i + ii) / 8 + 1).row(ti * 2 + m) + (tj * 2) * 8;

                    _mm256_store_ps(output0, _mm512_extractf32x8_ps(_tmp0, 0));
                    _mm256_store_ps(output1, _mm512_extractf32x8_ps(_tmp0, 1));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm256_store_ps(output0 + 8, _mm512_extractf32x8_ps(_tmp1, 0));
                        _mm256_store_ps(output1 + 8, _mm512_extractf32x8_ps(_tmp1, 1));
                    }
                }
                if (out_elempack == 4)
                {
                    float* output0 = top_blob.channel((i + ii) / 4).row(ti * 2 + m) + (tj * 2) * 4;
                    float* output1 = top_blob.channel((i + ii) / 4 + 1).row(ti * 2 + m) + (tj * 2) * 4;
                    float* output2 = top_blob.channel((i + ii) / 4 + 2).row(ti * 2 + m) + (tj * 2) * 4;
                    float* output3 = top_blob.channel((i + ii) / 4 + 3).row(ti * 2 + m) + (tj * 2) * 4;

                    _mm_store_ps(output0, _mm512_extractf32x4_ps(_tmp0, 0));
                    _mm_store_ps(output1, _mm512_extractf32x4_ps(_tmp0, 1));
                    _mm_store_ps(output2, _mm512_extractf32x4_ps(_tmp0, 2));
                    _mm_store_ps(output3, _mm512_extractf32x4_ps(_tmp0, 3));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm_store_ps(output0 + 4, _mm512_extractf32x4_ps(_tmp1, 0));
                        _mm_store_ps(output1 + 4, _mm512_extractf32x4_ps(_tmp1, 1));
                        _mm_store_ps(output2 + 4, _mm512_extractf32x4_ps(_tmp1, 2));
                        _mm_store_ps(output3 + 4, _mm512_extractf32x4_ps(_tmp1, 3));
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[16];
                    float tmp1[16];
                    _mm512_storeu_ps(tmp0, _tmp0);
                    _mm512_storeu_ps(tmp1, _tmp1);

                    float* output0 = top_blob.channel(i + ii).row(ti * 2 + m) + (tj * 2);
                    float* output1 = top_blob.channel(i + ii + 1).row(ti * 2 + m) + (tj * 2);
                    float* output2 = top_blob.channel(i + ii + 2).row(ti * 2 + m) + (tj * 2);
                    float* output3 = top_blob.channel(i + ii + 3).row(ti * 2 + m) + (tj * 2);
                    float* output4 = top_blob.channel(i + ii + 4).row(ti * 2 + m) + (tj * 2);
                    float* output5 = top_blob.channel(i + ii + 5).row(ti * 2 + m) + (tj * 2);
                    float* output6 = top_blob.channel(i + ii + 6).row(ti * 2 + m) + (tj * 2);
                    float* output7 = top_blob.channel(i + ii + 7).row(ti * 2 + m) + (tj * 2);
                    float* output8 = top_blob.channel(i + ii + 8).row(ti * 2 + m) + (tj * 2);
                    float* output9 = top_blob.channel(i + ii + 9).row(ti * 2 + m) + (tj * 2);
                    float* outputa = top_blob.channel(i + ii + 10).row(ti * 2 + m) + (tj * 2);
                    float* outputb = top_blob.channel(i + ii + 11).row(ti * 2 + m) + (tj * 2);
                    float* outputc = top_blob.channel(i + ii + 12).row(ti * 2 + m) + (tj * 2);
                    float* outputd = top_blob.channel(i + ii + 13).row(ti * 2 + m) + (tj * 2);
                    float* outpute = top_blob.channel(i + ii + 14).row(ti * 2 + m) + (tj * 2);
                    float* outputf = top_blob.channel(i + ii + 15).row(ti * 2 + m) + (tj * 2);

                    output0[0] = tmp0[0];
                    output1[0] = tmp0[1];
                    output2[0] = tmp0[2];
                    output3[0] = tmp0[3];
                    output4[0] = tmp0[4];
                    output5[0] = tmp0[5];
                    output6[0] = tmp0[6];
                    output7[0] = tmp0[7];
                    output8[0] = tmp0[8];
                    output9[0] = tmp0[9];
                    outputa[0] = tmp0[10];
                    outputb[0] = tmp0[11];
                    outputc[0] = tmp0[12];
                    outputd[0] = tmp0[13];
                    outpute[0] = tmp0[14];
                    outputf[0] = tmp0[15];

                    if (tj * 2 + 1 < outw)
                    {
                        output0[1] = tmp1[0];
                        output1[1] = tmp1[1];
                        output2[1] = tmp1[2];
                        output3[1] = tmp1[3];
                        output4[1] = tmp1[4];
                        output5[1] = tmp1[5];
                        output6[1] = tmp1[6];
                        output7[1] = tmp1[7];
                        output8[1] = tmp1[8];
                        output9[1] = tmp1[9];
                        outputa[1] = tmp1[10];
                        outputb[1] = tmp1[11];
                        outputc[1] = tmp1[12];
                        outputd[1] = tmp1[13];
                        outpute[1] = tmp1[14];
                        outputf[1] = tmp1[15];
                    }
                }
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _bias0 = biasptr ? _mm256_loadu_ps(biasptr + i + ii) : _mm256_setzero_ps();

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[2][4][8];

            for (int m = 0; m < 4; m++)
            {
                const float* r0 = top_tile.depth(m * 4).row(jj) + ii;
                const float* r1 = top_tile.depth(m * 4 + 1).row(jj) + ii;
                const float* r2 = top_tile.depth(m * 4 + 2).row(jj) + ii;
                const float* r3 = top_tile.depth(m * 4 + 3).row(jj) + ii;

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
            }
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
                    float* output0 = top_blob.channel((i + ii) / 8).row(ti * 2 + m) + (tj * 2) * 8;

                    _mm256_store_ps(output0, _tmp0);
                    if (tj * 2 + 1 < outw)
                    {
                        _mm256_store_ps(output0 + 8, _tmp1);
                    }
                }
                if (out_elempack == 4)
                {
                    float* output0 = top_blob.channel((i + ii) / 4).row(ti * 2 + m) + (tj * 2) * 4;
                    float* output1 = top_blob.channel((i + ii) / 4 + 1).row(ti * 2 + m) + (tj * 2) * 4;

                    _mm_store_ps(output0, _mm256_extractf128_ps(_tmp0, 0));
                    _mm_store_ps(output1, _mm256_extractf128_ps(_tmp0, 1));
                    if (tj * 2 + 1 < outw)
                    {
                        _mm_store_ps(output0 + 4, _mm256_extractf128_ps(_tmp1, 0));
                        _mm_store_ps(output1 + 4, _mm256_extractf128_ps(_tmp1, 1));
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    _mm256_storeu_ps(tmp0, _tmp0);
                    _mm256_storeu_ps(tmp1, _tmp1);

                    float* output0 = top_blob.channel(i + ii).row(ti * 2 + m) + (tj * 2);
                    float* output1 = top_blob.channel(i + ii + 1).row(ti * 2 + m) + (tj * 2);
                    float* output2 = top_blob.channel(i + ii + 2).row(ti * 2 + m) + (tj * 2);
                    float* output3 = top_blob.channel(i + ii + 3).row(ti * 2 + m) + (tj * 2);
                    float* output4 = top_blob.channel(i + ii + 4).row(ti * 2 + m) + (tj * 2);
                    float* output5 = top_blob.channel(i + ii + 5).row(ti * 2 + m) + (tj * 2);
                    float* output6 = top_blob.channel(i + ii + 6).row(ti * 2 + m) + (tj * 2);
                    float* output7 = top_blob.channel(i + ii + 7).row(ti * 2 + m) + (tj * 2);

                    output0[0] = tmp0[0];
                    output1[0] = tmp0[1];
                    output2[0] = tmp0[2];
                    output3[0] = tmp0[3];
                    output4[0] = tmp0[4];
                    output5[0] = tmp0[5];
                    output6[0] = tmp0[6];
                    output7[0] = tmp0[7];

                    if (tj * 2 + 1 < outw)
                    {
                        output0[1] = tmp1[0];
                        output1[1] = tmp1[1];
                        output2[1] = tmp1[2];
                        output3[1] = tmp1[3];
                        output4[1] = tmp1[4];
                        output5[1] = tmp1[5];
                        output6[1] = tmp1[6];
                        output7[1] = tmp1[7];
                    }
                }
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _bias0 = biasptr ? _mm_loadu_ps(biasptr + i + ii) : _mm_setzero_ps();

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(16))
#else
            __attribute__((aligned(16)))
#endif
            float tmp[2][4][4];

            for (int m = 0; m < 4; m++)
            {
                const float* r0 = top_tile.depth(m * 4).row(jj) + ii;
                const float* r1 = top_tile.depth(m * 4 + 1).row(jj) + ii;
                const float* r2 = top_tile.depth(m * 4 + 2).row(jj) + ii;
                const float* r3 = top_tile.depth(m * 4 + 3).row(jj) + ii;

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
            }
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
                    float* output0 = top_blob.channel((i + ii) / 4).row(ti * 2 + m) + (tj * 2) * 4;

                    _mm_store_ps(output0, _tmp0);
                    if (tj * 2 + 1 < outw) _mm_store_ps(output0 + 4, _tmp1);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    _mm_storeu_ps(tmp0, _tmp0);
                    _mm_storeu_ps(tmp1, _tmp1);

                    float* output0 = top_blob.channel(i + ii).row(ti * 2 + m) + (tj * 2);
                    float* output1 = top_blob.channel(i + ii + 1).row(ti * 2 + m) + (tj * 2);
                    float* output2 = top_blob.channel(i + ii + 2).row(ti * 2 + m) + (tj * 2);
                    float* output3 = top_blob.channel(i + ii + 3).row(ti * 2 + m) + (tj * 2);

                    output0[0] = tmp0[0];
                    output1[0] = tmp0[1];
                    output2[0] = tmp0[2];
                    output3[0] = tmp0[3];

                    if (tj * 2 + 1 < outw)
                    {
                        output0[1] = tmp1[0];
                        output1[1] = tmp1[1];
                        output2[1] = tmp1[2];
                        output3[1] = tmp1[3];
                    }
                }
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            float tmp[2][4][2];

            for (int m = 0; m < 4; m++)
            {
                float r00 = top_tile.depth(m * 4).row(jj)[ii];
                float r01 = top_tile.depth(m * 4).row(jj)[ii + 1];
                float r10 = top_tile.depth(m * 4 + 1).row(jj)[ii];
                float r11 = top_tile.depth(m * 4 + 1).row(jj)[ii + 1];
                float r20 = top_tile.depth(m * 4 + 2).row(jj)[ii];
                float r21 = top_tile.depth(m * 4 + 2).row(jj)[ii + 1];
                float r30 = top_tile.depth(m * 4 + 3).row(jj)[ii];
                float r31 = top_tile.depth(m * 4 + 3).row(jj)[ii + 1];

                float tmp00 = r00 + r10 + r20;
                float tmp01 = r01 + r11 + r21;
                float tmp10 = r10 - r20 + r30;
                float tmp11 = r11 - r21 + r31;

                tmp[0][m][0] = tmp00;
                tmp[0][m][1] = tmp01;
                tmp[1][m][0] = tmp10;
                tmp[1][m][1] = tmp11;
            }
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
                    float* output0 = top_blob.channel(i + ii).row(ti * 2 + m) + (tj * 2);
                    float* output1 = top_blob.channel(i + ii + 1).row(ti * 2 + m) + (tj * 2);

                    output0[0] = tmp00;
                    output1[0] = tmp01;
                    if (tj * 2 + 1 < outw)
                    {
                        output0[1] = tmp10;
                        output1[1] = tmp11;
                    }
                }
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            float tmp[2][4];

            for (int m = 0; m < 4; m++)
            {
                float r0 = top_tile.depth(m * 4).row(jj)[ii];
                float r1 = top_tile.depth(m * 4 + 1).row(jj)[ii];
                float r2 = top_tile.depth(m * 4 + 2).row(jj)[ii];
                float r3 = top_tile.depth(m * 4 + 3).row(jj)[ii];

                float tmp0 = r0 + r1 + r2;
                float tmp1 = r1 - r2 + r3;

                tmp[0][m] = tmp0;
                tmp[1][m] = tmp1;
            }
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
                    float* output0 = top_blob.channel(i + ii).row(ti * 2 + m) + (tj * 2);

                    output0[0] = tmp0;
                    if (tj * 2 + 1 < outw) output0[1] = tmp1;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, const Option& opt)
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

    int nT = opt.num_threads;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat B_tileX(B * TILE_N * TILE_K, 1, nT, 4u, opt.blob_allocator);
    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.blob_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_N; ppj++)
    {
        const int j = ppj * TILE_N;

        Mat B_tile = B_tileX.channel(get_omp_thread_num());

        const int max_jj = std::min((N - j), TILE_N);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd23_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk);
        }
    }

    Mat tmpX;
    if (TILE_K < K)
    {
        tmpX.create(TILE_M * TILE_N, B, nT, 4u, opt.blob_allocator);
    }

    Mat top_tileX(TILE_M, TILE_N, B, nT, 4u, opt.blob_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat tmp;
        if (K > TILE_K)
            tmp = tmpX.channel(get_omp_thread_num());

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

                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, tmp, B, max_ii, max_jj, k, max_kk, k_end);
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
            // const float ktm[6][3] = {
            //     {1.0f / 4, 0.0f, 0.0f},
            //     {-1.0f / 6, -1.0f / 6, -1.0f / 6},
            //     {-1.0f / 6, 1.0f / 6, -1.0f / 6},
            //     {1.0f / 24, 1.0f / 12, 1.0f / 6},
            //     {1.0f / 24, -1.0f / 12, 1.0f / 6},
            //     {0.0f, 0.0f, 1.0f}
            // };
            const float ktm0 = 1.0f / 4;
            const float ktm1 = 1.0f / 6;
            const float ktm2 = 1.0f / 12;
            const float ktm3 = 1.0f / 24;

            float tmp[6][3];

            for (int m = 0; m < 3; m++)
            {
                const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9 + m * 3;

                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0 * ktm0;
                tmp[1][m] = -r0 * ktm1 - r1 * ktm1 - r2 * ktm1;
                tmp[2][m] = -r0 * ktm1 + r1 * ktm1 - r2 * ktm1;
                tmp[3][m] = r0 * ktm3 + r1 * ktm2 + r2 * ktm1;
                tmp[4][m] = r0 * ktm3 - r1 * ktm2 + r2 * ktm1;
                tmp[5][m] = r2;
            }

            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0 * ktm0;
                float z1 = -r0 * ktm1 - r1 * ktm1 - r2 * ktm1;
                float z2 = -r0 * ktm1 + r1 * ktm1 - r2 * ktm1;
                float z3 = r0 * ktm3 + r1 * ktm2 + r2 * ktm1;
                float z4 = r0 * ktm3 - r1 * ktm2 + r2 * ktm1;
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

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, 4u, opt.blob_allocator);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 4u, opt.blob_allocator);

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

static inline void conv3x3s1_winograd43_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    // const float itm[6][6] = {
    //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
    //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
    //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
    //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
    //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
    //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;

    const int w_tiles = (w + 1) / 4;

    float* ptmp = B;

    int kk = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; kk + 15 < max_kk; kk += 16)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(64))
#else
            __attribute__((aligned(64)))
#endif
            float tmp[6][6][16];

            __m512 _vm5 = _mm512_set1_ps(-5.f);
            __m512 _vm4 = _mm512_set1_ps(-4.f);
            __m512 _v4 = _mm512_set1_ps(4.f);
            __m512 _vm2 = _mm512_set1_ps(-2.f);
            __m512 _v2 = _mm512_set1_ps(2.f);

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
                        const float* r0 = bottom_blob.channel((k + kk) / 16).row(ti * 4 + m) + (tj * 4) * 16;

                        _r0 = _mm512_load_ps(r0);
                        if (tj * 4 + 1 < w) _r1 = _mm512_load_ps(r0 + 16);
                        if (tj * 4 + 2 < w) _r2 = _mm512_load_ps(r0 + 32);
                        if (tj * 4 + 3 < w) _r3 = _mm512_load_ps(r0 + 48);
                        if (tj * 4 + 4 < w) _r4 = _mm512_load_ps(r0 + 64);
                        if (tj * 4 + 5 < w) _r5 = _mm512_load_ps(r0 + 80);
                    }
                    if (elempack == 8)
                    {
                        const float* r0 = bottom_blob.channel((k + kk) / 8).row(ti * 4 + m) + (tj * 4) * 8;
                        const float* r1 = bottom_blob.channel((k + kk) / 8 + 1).row(ti * 4 + m) + (tj * 4) * 8;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0)), _mm256_load_ps(r1), 1);
                        if (tj * 4 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 8)), _mm256_load_ps(r1 + 8), 1);
                        if (tj * 4 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 16)), _mm256_load_ps(r1 + 16), 1);
                        if (tj * 4 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 24)), _mm256_load_ps(r1 + 24), 1);
                        if (tj * 4 + 4 < w) _r4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 32)), _mm256_load_ps(r1 + 32), 1);
                        if (tj * 4 + 5 < w) _r5 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + 40)), _mm256_load_ps(r1 + 40), 1);
                    }
                    if (elempack == 4)
                    {
                        const float* r0 = bottom_blob.channel((k + kk) / 4).row(ti * 4 + m) + (tj * 4) * 4;
                        const float* r1 = bottom_blob.channel((k + kk) / 4 + 1).row(ti * 4 + m) + (tj * 4) * 4;
                        const float* r2 = bottom_blob.channel((k + kk) / 4 + 2).row(ti * 4 + m) + (tj * 4) * 4;
                        const float* r3 = bottom_blob.channel((k + kk) / 4 + 3).row(ti * 4 + m) + (tj * 4) * 4;

                        _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2)), _mm_load_ps(r3), 1), 1);
                        if (tj * 4 + 1 < w) _r1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 4)), _mm_load_ps(r3 + 4), 1), 1);
                        if (tj * 4 + 2 < w) _r2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 8)), _mm_load_ps(r3 + 8), 1), 1);
                        if (tj * 4 + 3 < w) _r3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 12)), _mm_load_ps(r3 + 12), 1), 1);
                        if (tj * 4 + 4 < w) _r4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 16)), _mm_load_ps(r1 + 16), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 16)), _mm_load_ps(r3 + 16), 1), 1);
                        if (tj * 4 + 5 < w) _r5 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 20)), _mm_load_ps(r1 + 20), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + 20)), _mm_load_ps(r3 + 20), 1), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r0 = bottom_blob.channel(k + kk).row(ti * 4 + m) + (tj * 4);
                        const float* r1 = bottom_blob.channel(k + kk + 1).row(ti * 4 + m) + (tj * 4);
                        const float* r2 = bottom_blob.channel(k + kk + 2).row(ti * 4 + m) + (tj * 4);
                        const float* r3 = bottom_blob.channel(k + kk + 3).row(ti * 4 + m) + (tj * 4);
                        const float* r4 = bottom_blob.channel(k + kk + 4).row(ti * 4 + m) + (tj * 4);
                        const float* r5 = bottom_blob.channel(k + kk + 5).row(ti * 4 + m) + (tj * 4);
                        const float* r6 = bottom_blob.channel(k + kk + 6).row(ti * 4 + m) + (tj * 4);
                        const float* r7 = bottom_blob.channel(k + kk + 7).row(ti * 4 + m) + (tj * 4);
                        const float* r8 = bottom_blob.channel(k + kk + 8).row(ti * 4 + m) + (tj * 4);
                        const float* r9 = bottom_blob.channel(k + kk + 9).row(ti * 4 + m) + (tj * 4);
                        const float* ra = bottom_blob.channel(k + kk + 10).row(ti * 4 + m) + (tj * 4);
                        const float* rb = bottom_blob.channel(k + kk + 11).row(ti * 4 + m) + (tj * 4);
                        const float* rc = bottom_blob.channel(k + kk + 12).row(ti * 4 + m) + (tj * 4);
                        const float* rd = bottom_blob.channel(k + kk + 13).row(ti * 4 + m) + (tj * 4);
                        const float* re = bottom_blob.channel(k + kk + 14).row(ti * 4 + m) + (tj * 4);
                        const float* rf = bottom_blob.channel(k + kk + 15).row(ti * 4 + m) + (tj * 4);

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
                        if (tj * 4 + 4 < w)
                        {
                            float tmp[16] = {r0[4], r1[4], r2[4], r3[4], r4[4], r5[4], r6[4], r7[4], r8[4], r9[4], ra[4], rb[4], rc[4], rd[4], re[4], rf[4]};
                            _r4 = _mm512_loadu_ps(tmp);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            float tmp[16] = {r0[5], r1[5], r2[5], r3[5], r4[5], r5[5], r6[5], r7[5], r8[5], r9[5], ra[5], rb[5], rc[5], rd[5], re[5], rf[5]};
                            _r5 = _mm512_loadu_ps(tmp);
                        }
                    }
                }

                __m512 _tmp0 = _mm512_fmadd_ps(_vm5, _r2, _mm512_fmadd_ps(_v4, _r0, _r4));
                __m512 _tmp1 = _mm512_fmadd_ps(_vm4, _mm512_add_ps(_r1, _r2), _mm512_add_ps(_r4, _r3));
                __m512 _tmp2 = _mm512_fmadd_ps(_v4, _mm512_sub_ps(_r1, _r2), _mm512_sub_ps(_r4, _r3));
                __m512 _tmp3 = _mm512_fmadd_ps(_vm2, _mm512_sub_ps(_r1, _r3), _mm512_sub_ps(_r4, _r2));
                __m512 _tmp4 = _mm512_fmadd_ps(_v2, _mm512_sub_ps(_r1, _r3), _mm512_sub_ps(_r4, _r2));
                __m512 _tmp5 = _mm512_fmadd_ps(_vm5, _r3, _mm512_fmadd_ps(_v4, _r1, _r5));

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);
                _mm512_store_ps(tmp[2][m], _tmp2);
                _mm512_store_ps(tmp[3][m], _tmp3);
                _mm512_store_ps(tmp[4][m], _tmp4);
                _mm512_store_ps(tmp[5][m], _tmp5);
            }
            for (int m = 0; m < 6; m++)
            {
                __m512 _r0 = _mm512_load_ps(tmp[m][0]);
                __m512 _r1 = _mm512_load_ps(tmp[m][1]);
                __m512 _r2 = _mm512_load_ps(tmp[m][2]);
                __m512 _r3 = _mm512_load_ps(tmp[m][3]);
                __m512 _r4 = _mm512_load_ps(tmp[m][4]);
                __m512 _r5 = _mm512_load_ps(tmp[m][5]);

                __m512 _tmp0 = _mm512_fmadd_ps(_vm5, _r2, _mm512_fmadd_ps(_v4, _r0, _r4));
                __m512 _tmp1 = _mm512_fmadd_ps(_vm4, _mm512_add_ps(_r1, _r2), _mm512_add_ps(_r4, _r3));
                __m512 _tmp2 = _mm512_fmadd_ps(_v4, _mm512_sub_ps(_r1, _r2), _mm512_sub_ps(_r4, _r3));
                __m512 _tmp3 = _mm512_fmadd_ps(_vm2, _mm512_sub_ps(_r1, _r3), _mm512_sub_ps(_r4, _r2));
                __m512 _tmp4 = _mm512_fmadd_ps(_v2, _mm512_sub_ps(_r1, _r3), _mm512_sub_ps(_r4, _r2));
                __m512 _tmp5 = _mm512_fmadd_ps(_vm5, _r3, _mm512_fmadd_ps(_v4, _r1, _r5));

                _mm512_store_ps(ptmp, _tmp0);
                _mm512_store_ps(ptmp + 16, _tmp1);
                _mm512_store_ps(ptmp + 32, _tmp2);
                _mm512_store_ps(ptmp + 48, _tmp3);
                _mm512_store_ps(ptmp + 64, _tmp4);
                _mm512_store_ps(ptmp + 80, _tmp5);
                ptmp += 96;
            }
        }
    }
#endif // __AVX512F__
    for (; kk + 7 < max_kk; kk += 8)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[6][6][8];

            __m256 _vm5 = _mm256_set1_ps(-5.f);
            __m256 _vm4 = _mm256_set1_ps(-4.f);
            __m256 _v4 = _mm256_set1_ps(4.f);
            __m256 _vm2 = _mm256_set1_ps(-2.f);
            __m256 _v2 = _mm256_set1_ps(2.f);

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
                        const float* r0 = bottom_blob.channel((k + kk) / 8).row(ti * 4 + m) + (tj * 4) * 8;

                        _r0 = _mm256_load_ps(r0);
                        if (tj * 4 + 1 < w) _r1 = _mm256_load_ps(r0 + 8);
                        if (tj * 4 + 2 < w) _r2 = _mm256_load_ps(r0 + 16);
                        if (tj * 4 + 3 < w) _r3 = _mm256_load_ps(r0 + 24);
                        if (tj * 4 + 4 < w) _r4 = _mm256_load_ps(r0 + 32);
                        if (tj * 4 + 5 < w) _r5 = _mm256_load_ps(r0 + 40);
                    }
                    if (elempack == 4)
                    {
                        const float* r0 = bottom_blob.channel((k + kk) / 4).row(ti * 4 + m) + (tj * 4) * 4;
                        const float* r1 = bottom_blob.channel((k + kk) / 4 + 1).row(ti * 4 + m) + (tj * 4) * 4;

                        _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1);
                        if (tj * 4 + 1 < w) _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 4)), _mm_load_ps(r1 + 4), 1);
                        if (tj * 4 + 2 < w) _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 8)), _mm_load_ps(r1 + 8), 1);
                        if (tj * 4 + 3 < w) _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 12)), _mm_load_ps(r1 + 12), 1);
                        if (tj * 4 + 4 < w) _r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 16)), _mm_load_ps(r1 + 16), 1);
                        if (tj * 4 + 5 < w) _r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + 20)), _mm_load_ps(r1 + 20), 1);
                    }
                    if (elempack == 1)
                    {
                        const float* r0 = bottom_blob.channel(k + kk).row(ti * 4 + m) + (tj * 4);
                        const float* r1 = bottom_blob.channel(k + kk + 1).row(ti * 4 + m) + (tj * 4);
                        const float* r2 = bottom_blob.channel(k + kk + 2).row(ti * 4 + m) + (tj * 4);
                        const float* r3 = bottom_blob.channel(k + kk + 3).row(ti * 4 + m) + (tj * 4);
                        const float* r4 = bottom_blob.channel(k + kk + 4).row(ti * 4 + m) + (tj * 4);
                        const float* r5 = bottom_blob.channel(k + kk + 5).row(ti * 4 + m) + (tj * 4);
                        const float* r6 = bottom_blob.channel(k + kk + 6).row(ti * 4 + m) + (tj * 4);
                        const float* r7 = bottom_blob.channel(k + kk + 7).row(ti * 4 + m) + (tj * 4);

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
                        if (tj * 4 + 4 < w)
                        {
                            float tmp[8] = {r0[4], r1[4], r2[4], r3[4], r4[4], r5[4], r6[4], r7[4]};
                            _r4 = _mm256_loadu_ps(tmp);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            float tmp[8] = {r0[5], r1[5], r2[5], r3[5], r4[5], r5[5], r6[5], r7[5]};
                            _r5 = _mm256_loadu_ps(tmp);
                        }
                    }
                }

                __m256 _tmp0 = _mm256_comp_fmadd_ps(_vm5, _r2, _mm256_comp_fmadd_ps(_v4, _r0, _r4));
                __m256 _tmp1 = _mm256_comp_fmadd_ps(_vm4, _mm256_add_ps(_r1, _r2), _mm256_add_ps(_r4, _r3));
                __m256 _tmp2 = _mm256_comp_fmadd_ps(_v4, _mm256_sub_ps(_r1, _r2), _mm256_sub_ps(_r4, _r3));
                __m256 _tmp3 = _mm256_comp_fmadd_ps(_vm2, _mm256_sub_ps(_r1, _r3), _mm256_sub_ps(_r4, _r2));
                __m256 _tmp4 = _mm256_comp_fmadd_ps(_v2, _mm256_sub_ps(_r1, _r3), _mm256_sub_ps(_r4, _r2));
                __m256 _tmp5 = _mm256_comp_fmadd_ps(_vm5, _r3, _mm256_comp_fmadd_ps(_v4, _r1, _r5));

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
            }
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

                __m256 _tmp0 = _mm256_comp_fmadd_ps(_vm5, _r2, _mm256_comp_fmadd_ps(_v4, _r0, _r4));
                __m256 _tmp1 = _mm256_comp_fmadd_ps(_vm4, _mm256_add_ps(_r1, _r2), _mm256_add_ps(_r4, _r3));
                __m256 _tmp2 = _mm256_comp_fmadd_ps(_v4, _mm256_sub_ps(_r1, _r2), _mm256_sub_ps(_r4, _r3));
                __m256 _tmp3 = _mm256_comp_fmadd_ps(_vm2, _mm256_sub_ps(_r1, _r3), _mm256_sub_ps(_r4, _r2));
                __m256 _tmp4 = _mm256_comp_fmadd_ps(_v2, _mm256_sub_ps(_r1, _r3), _mm256_sub_ps(_r4, _r2));
                __m256 _tmp5 = _mm256_comp_fmadd_ps(_vm5, _r3, _mm256_comp_fmadd_ps(_v4, _r1, _r5));

                _mm256_store_ps(ptmp, _tmp0);
                _mm256_store_ps(ptmp + 8, _tmp1);
                _mm256_store_ps(ptmp + 16, _tmp2);
                _mm256_store_ps(ptmp + 24, _tmp3);
                _mm256_store_ps(ptmp + 32, _tmp4);
                _mm256_store_ps(ptmp + 40, _tmp5);
                ptmp += 48;
            }
        }
    }
#endif // __AVX__
    for (; kk + 3 < max_kk; kk += 4)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(16))
#else
            __attribute__((aligned(16)))
#endif
            float tmp[6][6][4];

            __m128 _vm5 = _mm_set1_ps(-5.f);
            __m128 _vm4 = _mm_set1_ps(-4.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _vm2 = _mm_set1_ps(-2.f);
            __m128 _v2 = _mm_set1_ps(2.f);

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
                        const float* r0 = bottom_blob.channel((k + kk) / 4).row(ti * 4 + m) + (tj * 4) * 4;

                        _r0 = _mm_load_ps(r0);
                        if (tj * 4 + 1 < w) _r1 = _mm_load_ps(r0 + 4);
                        if (tj * 4 + 2 < w) _r2 = _mm_load_ps(r0 + 8);
                        if (tj * 4 + 3 < w) _r3 = _mm_load_ps(r0 + 12);
                        if (tj * 4 + 4 < w) _r4 = _mm_load_ps(r0 + 16);
                        if (tj * 4 + 5 < w) _r5 = _mm_load_ps(r0 + 20);
                    }
                    if (elempack == 1)
                    {
                        const float* r0 = bottom_blob.channel(k + kk).row(ti * 4 + m) + (tj * 4);
                        const float* r1 = bottom_blob.channel(k + kk + 1).row(ti * 4 + m) + (tj * 4);
                        const float* r2 = bottom_blob.channel(k + kk + 2).row(ti * 4 + m) + (tj * 4);
                        const float* r3 = bottom_blob.channel(k + kk + 3).row(ti * 4 + m) + (tj * 4);

                        __m128 _t0 = _mm_loadu_ps(r0);
                        __m128 _t1 = _mm_loadu_ps(r1);
                        __m128 _t2 = _mm_loadu_ps(r2);
                        __m128 _t3 = _mm_loadu_ps(r3);

                        _MM_TRANSPOSE4_PS(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 4 + 1 < w) _r1 = _t1;
                        if (tj * 4 + 2 < w) _r2 = _t2;
                        if (tj * 4 + 3 < w) _r3 = _t3;
                        if (tj * 4 + 4 < w)
                        {
                            float tmp[4] = {r0[4], r1[4], r2[4], r3[4]};
                            _r4 = _mm_loadu_ps(tmp);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            float tmp[4] = {r0[5], r1[5], r2[5], r3[5]};
                            _r5 = _mm_loadu_ps(tmp);
                        }
                    }
                }

                __m128 _tmp0 = _mm_comp_fmadd_ps(_vm5, _r2, _mm_comp_fmadd_ps(_v4, _r0, _r4));
                __m128 _tmp1 = _mm_comp_fmadd_ps(_vm4, _mm_add_ps(_r1, _r2), _mm_add_ps(_r4, _r3));
                __m128 _tmp2 = _mm_comp_fmadd_ps(_v4, _mm_sub_ps(_r1, _r2), _mm_sub_ps(_r4, _r3));
                __m128 _tmp3 = _mm_comp_fmadd_ps(_vm2, _mm_sub_ps(_r1, _r3), _mm_sub_ps(_r4, _r2));
                __m128 _tmp4 = _mm_comp_fmadd_ps(_v2, _mm_sub_ps(_r1, _r3), _mm_sub_ps(_r4, _r2));
                __m128 _tmp5 = _mm_comp_fmadd_ps(_vm5, _r3, _mm_comp_fmadd_ps(_v4, _r1, _r5));

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
            }
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

                __m128 _tmp0 = _mm_comp_fmadd_ps(_vm5, _r2, _mm_comp_fmadd_ps(_v4, _r0, _r4));
                __m128 _tmp1 = _mm_comp_fmadd_ps(_vm4, _mm_add_ps(_r1, _r2), _mm_add_ps(_r4, _r3));
                __m128 _tmp2 = _mm_comp_fmadd_ps(_v4, _mm_sub_ps(_r1, _r2), _mm_sub_ps(_r4, _r3));
                __m128 _tmp3 = _mm_comp_fmadd_ps(_vm2, _mm_sub_ps(_r1, _r3), _mm_sub_ps(_r4, _r2));
                __m128 _tmp4 = _mm_comp_fmadd_ps(_v2, _mm_sub_ps(_r1, _r3), _mm_sub_ps(_r4, _r2));
                __m128 _tmp5 = _mm_comp_fmadd_ps(_vm5, _r3, _mm_comp_fmadd_ps(_v4, _r1, _r5));

                _mm_store_ps(ptmp, _tmp0);
                _mm_store_ps(ptmp + 4, _tmp1);
                _mm_store_ps(ptmp + 8, _tmp2);
                _mm_store_ps(ptmp + 12, _tmp3);
                _mm_store_ps(ptmp + 16, _tmp4);
                _mm_store_ps(ptmp + 20, _tmp5);
                ptmp += 24;
            }
        }
    }
#endif // __SSE2__
    for (; kk + 1 < max_kk; kk += 2)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            float tmp[6][6][2];

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
                        const float* r0123_0 = bottom_blob.channel(k + kk).row(ti * 4 + m) + (tj * 4);
                        const float* r0123_1 = bottom_blob.channel(k + kk + 1).row(ti * 4 + m) + (tj * 4);

                        r00 = r0123_0[0];
                        r01 = r0123_1[0];
                        if (tj * 4 + 1 < w)
                        {
                            r10 = r0123_0[1];
                            r11 = r0123_1[1];
                        }
                        if (tj * 4 + 2 < w)
                        {
                            r20 = r0123_0[2];
                            r21 = r0123_1[2];
                        }
                        if (tj * 4 + 3 < w)
                        {
                            r30 = r0123_0[3];
                            r31 = r0123_1[3];
                        }
                        if (tj * 4 + 4 < w)
                        {
                            r40 = r0123_0[4];
                            r41 = r0123_1[4];
                        }
                        if (tj * 4 + 5 < w)
                        {
                            r50 = r0123_0[5];
                            r51 = r0123_1[5];
                        }
                    }
                }

                tmp[0][m][0] = r00 * 4.f - r20 * 5.f + r40;
                tmp[0][m][1] = r01 * 4.f - r21 * 5.f + r41;
                tmp[1][m][0] = -r10 * 4.f - r20 * 4.f + r30 + r40;
                tmp[1][m][1] = -r11 * 4.f - r21 * 4.f + r31 + r41;
                tmp[2][m][0] = r10 * 4.f - r20 * 4.f - r30 + r40;
                tmp[2][m][1] = r11 * 4.f - r21 * 4.f - r31 + r41;
                tmp[3][m][0] = -r10 * 2.f - r20 + r30 * 2.f + r40;
                tmp[3][m][1] = -r11 * 2.f - r21 + r31 * 2.f + r41;
                tmp[4][m][0] = r10 * 2.f - r20 - r30 * 2.f + r40;
                tmp[4][m][1] = r11 * 2.f - r21 - r31 * 2.f + r41;
                tmp[5][m][0] = r10 * 4.f - r30 * 5.f + r50;
                tmp[5][m][1] = r11 * 4.f - r31 * 5.f + r51;
            }
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

                float z00 = r00 * 4.f - r20 * 5.f + r40;
                float z01 = r01 * 4.f - r21 * 5.f + r41;
                float z10 = -r10 * 4.f - r20 * 4.f + r30 + r40;
                float z11 = -r11 * 4.f - r21 * 4.f + r31 + r41;
                float z20 = r10 * 4.f - r20 * 4.f - r30 + r40;
                float z21 = r11 * 4.f - r21 * 4.f - r31 + r41;
                float z30 = -r10 * 2.f - r20 + r30 * 2.f + r40;
                float z31 = -r11 * 2.f - r21 + r31 * 2.f + r41;
                float z40 = r10 * 2.f - r20 - r30 * 2.f + r40;
                float z41 = r11 * 2.f - r21 - r31 * 2.f + r41;
                float z50 = r10 * 4.f - r30 * 5.f + r50;
                float z51 = r11 * 4.f - r31 * 5.f + r51;

                ptmp[0] = z00;
                ptmp[1] = z01;
                ptmp[2] = z10;
                ptmp[3] = z11;
                ptmp[4] = z20;
                ptmp[5] = z21;
                ptmp[6] = z30;
                ptmp[7] = z31;
                ptmp[8] = z40;
                ptmp[9] = z41;
                ptmp[10] = z50;
                ptmp[11] = z51;
                ptmp += 12;
            }
        }
    }
    for (; kk < max_kk; kk++)
    {
        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            float tmp[6][6];

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
                        const float* r0123 = bottom_blob.channel(k + kk).row(ti * 4 + m) + (tj * 4);

                        r0 = r0123[0];
                        if (tj * 4 + 1 < w) r1 = r0123[1];
                        if (tj * 4 + 2 < w) r2 = r0123[2];
                        if (tj * 4 + 3 < w) r3 = r0123[3];
                        if (tj * 4 + 4 < w) r4 = r0123[4];
                        if (tj * 4 + 5 < w) r5 = r0123[5];
                    }
                }

                tmp[0][m] = r0 * 4.f - r2 * 5.f + r4;
                tmp[1][m] = -r1 * 4.f - r2 * 4.f + r3 + r4;
                tmp[2][m] = r1 * 4.f - r2 * 4.f - r3 + r4;
                tmp[3][m] = -r1 * 2.f - r2 + r3 * 2.f + r4;
                tmp[4][m] = r1 * 2.f - r2 - r3 * 2.f + r4;
                tmp[5][m] = r1 * 4.f - r3 * 5.f + r5;
            }
            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float z0 = r0 * 4.f - r2 * 5.f + r4;
                float z1 = -r1 * 4.f - r2 * 4.f + r3 + r4;
                float z2 = r1 * 4.f - r2 * 4.f - r3 + r4;
                float z3 = -r1 * 2.f - r2 + r3 * 2.f + r4;
                float z4 = r1 * 2.f - r2 - r3 * 2.f + r4;
                float z5 = r1 * 4.f - r3 * 5.f + r5;

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

static inline void conv3x3s1_winograd43_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[4][6] = {
    //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;

    const int w_tiles = (outw + 3) / 4;

    const float* biasptr = bias;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        __m512 _bias0 = biasptr ? _mm512_loadu_ps(biasptr + i + ii) : _mm512_setzero_ps();

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(64))
#else
            __attribute__((aligned(64)))
#endif
            float tmp[4][6][16];

            __m512 _v2 = _mm512_set1_ps(2.f);
            __m512 _v4 = _mm512_set1_ps(4.f);
            __m512 _v8 = _mm512_set1_ps(8.f);

            for (int m = 0; m < 6; m++)
            {
                const float* r0 = top_tile.depth(m * 6).row(jj) + ii;
                const float* r1 = top_tile.depth(m * 6 + 1).row(jj) + ii;
                const float* r2 = top_tile.depth(m * 6 + 2).row(jj) + ii;
                const float* r3 = top_tile.depth(m * 6 + 3).row(jj) + ii;
                const float* r4 = top_tile.depth(m * 6 + 4).row(jj) + ii;
                const float* r5 = top_tile.depth(m * 6 + 5).row(jj) + ii;

                __m512 _r0 = _mm512_load_ps(r0);
                __m512 _r1 = _mm512_load_ps(r1);
                __m512 _r2 = _mm512_load_ps(r2);
                __m512 _r3 = _mm512_load_ps(r3);
                __m512 _r4 = _mm512_load_ps(r4);
                __m512 _r5 = _mm512_load_ps(r5);

                __m512 _tmp02a = _mm512_add_ps(_r1, _r2);
                __m512 _tmp13a = _mm512_sub_ps(_r1, _r2);
                __m512 _tmp02b = _mm512_add_ps(_r3, _r4);
                __m512 _tmp13b = _mm512_sub_ps(_r3, _r4);
                __m512 _tmp0 = _mm512_add_ps(_mm512_add_ps(_r0, _tmp02a), _tmp02b);
                __m512 _tmp1 = _mm512_fmadd_ps(_v2, _tmp13b, _tmp13a);
                __m512 _tmp2 = _mm512_fmadd_ps(_v4, _tmp02b, _tmp02a);
                __m512 _tmp3 = _mm512_fmadd_ps(_v8, _tmp13b, _mm512_add_ps(_r5, _tmp13a));

                _mm512_store_ps(tmp[0][m], _tmp0);
                _mm512_store_ps(tmp[1][m], _tmp1);
                _mm512_store_ps(tmp[2][m], _tmp2);
                _mm512_store_ps(tmp[3][m], _tmp3);
            }
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
                __m512 _tmp13a = _mm512_sub_ps(_r1, _r2);
                __m512 _tmp02b = _mm512_add_ps(_r3, _r4);
                __m512 _tmp13b = _mm512_sub_ps(_r3, _r4);
                __m512 _tmp0 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_add_ps(_r0, _tmp02a), _tmp02b));
                __m512 _tmp1 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_v2, _tmp13b, _tmp13a));
                __m512 _tmp2 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_v4, _tmp02b, _tmp02a));
                __m512 _tmp3 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_v8, _tmp13b, _mm512_add_ps(_r5, _tmp13a)));

                if (out_elempack == 16)
                {
                    float* output0 = top_blob.channel((i + ii) / 16).row(ti * 4 + m) + (tj * 4) * 16;

                    _mm512_store_ps(output0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm512_store_ps(output0 + 16, _tmp1);
                    if (tj * 4 + 2 < outw) _mm512_store_ps(output0 + 32, _tmp2);
                    if (tj * 4 + 3 < outw) _mm512_store_ps(output0 + 48, _tmp3);
                }
                if (out_elempack == 8)
                {
                    float* output0 = top_blob.channel((i + ii) / 8).row(ti * 4 + m) + (tj * 4) * 8;
                    float* output1 = top_blob.channel((i + ii) / 8 + 1).row(ti * 4 + m) + (tj * 4) * 8;

                    _mm256_store_ps(output0, _mm512_extractf32x8_ps(_tmp0, 0));
                    _mm256_store_ps(output1, _mm512_extractf32x8_ps(_tmp0, 1));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm256_store_ps(output0 + 8, _mm512_extractf32x8_ps(_tmp1, 0));
                        _mm256_store_ps(output1 + 8, _mm512_extractf32x8_ps(_tmp1, 1));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm256_store_ps(output0 + 16, _mm512_extractf32x8_ps(_tmp2, 0));
                        _mm256_store_ps(output1 + 16, _mm512_extractf32x8_ps(_tmp2, 1));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm256_store_ps(output0 + 24, _mm512_extractf32x8_ps(_tmp3, 0));
                        _mm256_store_ps(output1 + 24, _mm512_extractf32x8_ps(_tmp3, 1));
                    }
                }
                if (out_elempack == 4)
                {
                    float* output0 = top_blob.channel((i + ii) / 4).row(ti * 4 + m) + (tj * 4) * 4;
                    float* output1 = top_blob.channel((i + ii) / 4 + 1).row(ti * 4 + m) + (tj * 4) * 4;
                    float* output2 = top_blob.channel((i + ii) / 4 + 2).row(ti * 4 + m) + (tj * 4) * 4;
                    float* output3 = top_blob.channel((i + ii) / 4 + 3).row(ti * 4 + m) + (tj * 4) * 4;

                    _mm_store_ps(output0, _mm512_extractf32x4_ps(_tmp0, 0));
                    _mm_store_ps(output1, _mm512_extractf32x4_ps(_tmp0, 1));
                    _mm_store_ps(output2, _mm512_extractf32x4_ps(_tmp0, 2));
                    _mm_store_ps(output3, _mm512_extractf32x4_ps(_tmp0, 3));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm_store_ps(output0 + 4, _mm512_extractf32x4_ps(_tmp1, 0));
                        _mm_store_ps(output1 + 4, _mm512_extractf32x4_ps(_tmp1, 1));
                        _mm_store_ps(output2 + 4, _mm512_extractf32x4_ps(_tmp1, 2));
                        _mm_store_ps(output3 + 4, _mm512_extractf32x4_ps(_tmp1, 3));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm_store_ps(output0 + 8, _mm512_extractf32x4_ps(_tmp2, 0));
                        _mm_store_ps(output1 + 8, _mm512_extractf32x4_ps(_tmp2, 1));
                        _mm_store_ps(output2 + 8, _mm512_extractf32x4_ps(_tmp2, 2));
                        _mm_store_ps(output3 + 8, _mm512_extractf32x4_ps(_tmp2, 3));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm_store_ps(output0 + 12, _mm512_extractf32x4_ps(_tmp3, 0));
                        _mm_store_ps(output1 + 12, _mm512_extractf32x4_ps(_tmp3, 1));
                        _mm_store_ps(output2 + 12, _mm512_extractf32x4_ps(_tmp3, 2));
                        _mm_store_ps(output3 + 12, _mm512_extractf32x4_ps(_tmp3, 3));
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

                    float* output0 = top_blob.channel(i + ii).row(ti * 4 + m) + (tj * 4);
                    float* output1 = top_blob.channel(i + ii + 1).row(ti * 4 + m) + (tj * 4);
                    float* output2 = top_blob.channel(i + ii + 2).row(ti * 4 + m) + (tj * 4);
                    float* output3 = top_blob.channel(i + ii + 3).row(ti * 4 + m) + (tj * 4);
                    float* output4 = top_blob.channel(i + ii + 4).row(ti * 4 + m) + (tj * 4);
                    float* output5 = top_blob.channel(i + ii + 5).row(ti * 4 + m) + (tj * 4);
                    float* output6 = top_blob.channel(i + ii + 6).row(ti * 4 + m) + (tj * 4);
                    float* output7 = top_blob.channel(i + ii + 7).row(ti * 4 + m) + (tj * 4);
                    float* output8 = top_blob.channel(i + ii + 8).row(ti * 4 + m) + (tj * 4);
                    float* output9 = top_blob.channel(i + ii + 9).row(ti * 4 + m) + (tj * 4);
                    float* outputa = top_blob.channel(i + ii + 10).row(ti * 4 + m) + (tj * 4);
                    float* outputb = top_blob.channel(i + ii + 11).row(ti * 4 + m) + (tj * 4);
                    float* outputc = top_blob.channel(i + ii + 12).row(ti * 4 + m) + (tj * 4);
                    float* outputd = top_blob.channel(i + ii + 13).row(ti * 4 + m) + (tj * 4);
                    float* outpute = top_blob.channel(i + ii + 14).row(ti * 4 + m) + (tj * 4);
                    float* outputf = top_blob.channel(i + ii + 15).row(ti * 4 + m) + (tj * 4);

                    output0[0] = tmp0[0];
                    output1[0] = tmp0[1];
                    output2[0] = tmp0[2];
                    output3[0] = tmp0[3];
                    output4[0] = tmp0[4];
                    output5[0] = tmp0[5];
                    output6[0] = tmp0[6];
                    output7[0] = tmp0[7];
                    output8[0] = tmp0[8];
                    output9[0] = tmp0[9];
                    outputa[0] = tmp0[10];
                    outputb[0] = tmp0[11];
                    outputc[0] = tmp0[12];
                    outputd[0] = tmp0[13];
                    outpute[0] = tmp0[14];
                    outputf[0] = tmp0[15];
                    if (tj * 4 + 1 < outw)
                    {
                        output0[1] = tmp1[0];
                        output1[1] = tmp1[1];
                        output2[1] = tmp1[2];
                        output3[1] = tmp1[3];
                        output4[1] = tmp1[4];
                        output5[1] = tmp1[5];
                        output6[1] = tmp1[6];
                        output7[1] = tmp1[7];
                        output8[1] = tmp1[8];
                        output9[1] = tmp1[9];
                        outputa[1] = tmp1[10];
                        outputb[1] = tmp1[11];
                        outputc[1] = tmp1[12];
                        outputd[1] = tmp1[13];
                        outpute[1] = tmp1[14];
                        outputf[1] = tmp1[15];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        output0[2] = tmp2[0];
                        output1[2] = tmp2[1];
                        output2[2] = tmp2[2];
                        output3[2] = tmp2[3];
                        output4[2] = tmp2[4];
                        output5[2] = tmp2[5];
                        output6[2] = tmp2[6];
                        output7[2] = tmp2[7];
                        output8[2] = tmp2[8];
                        output9[2] = tmp2[9];
                        outputa[2] = tmp2[10];
                        outputb[2] = tmp2[11];
                        outputc[2] = tmp2[12];
                        outputd[2] = tmp2[13];
                        outpute[2] = tmp2[14];
                        outputf[2] = tmp2[15];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        output0[3] = tmp3[0];
                        output1[3] = tmp3[1];
                        output2[3] = tmp3[2];
                        output3[3] = tmp3[3];
                        output4[3] = tmp3[4];
                        output5[3] = tmp3[5];
                        output6[3] = tmp3[6];
                        output7[3] = tmp3[7];
                        output8[3] = tmp3[8];
                        output9[3] = tmp3[9];
                        outputa[3] = tmp3[10];
                        outputb[3] = tmp3[11];
                        outputc[3] = tmp3[12];
                        outputd[3] = tmp3[13];
                        outpute[3] = tmp3[14];
                        outputf[3] = tmp3[15];
                    }
                }
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _bias0 = biasptr ? _mm256_loadu_ps(biasptr + i + ii) : _mm256_setzero_ps();

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[4][6][8];

            __m256 _v2 = _mm256_set1_ps(2.f);
            __m256 _v4 = _mm256_set1_ps(4.f);
            __m256 _v8 = _mm256_set1_ps(8.f);

            for (int m = 0; m < 6; m++)
            {
                const float* r0 = top_tile.depth(m * 6).row(jj) + ii;
                const float* r1 = top_tile.depth(m * 6 + 1).row(jj) + ii;
                const float* r2 = top_tile.depth(m * 6 + 2).row(jj) + ii;
                const float* r3 = top_tile.depth(m * 6 + 3).row(jj) + ii;
                const float* r4 = top_tile.depth(m * 6 + 4).row(jj) + ii;
                const float* r5 = top_tile.depth(m * 6 + 5).row(jj) + ii;

                __m256 _r0 = _mm256_load_ps(r0);
                __m256 _r1 = _mm256_load_ps(r1);
                __m256 _r2 = _mm256_load_ps(r2);
                __m256 _r3 = _mm256_load_ps(r3);
                __m256 _r4 = _mm256_load_ps(r4);
                __m256 _r5 = _mm256_load_ps(r5);

                __m256 _tmp02a = _mm256_add_ps(_r1, _r2);
                __m256 _tmp13a = _mm256_sub_ps(_r1, _r2);
                __m256 _tmp02b = _mm256_add_ps(_r3, _r4);
                __m256 _tmp13b = _mm256_sub_ps(_r3, _r4);
                __m256 _tmp0 = _mm256_add_ps(_mm256_add_ps(_r0, _tmp02a), _tmp02b);
                __m256 _tmp1 = _mm256_comp_fmadd_ps(_v2, _tmp13b, _tmp13a);
                __m256 _tmp2 = _mm256_comp_fmadd_ps(_v4, _tmp02b, _tmp02a);
                __m256 _tmp3 = _mm256_comp_fmadd_ps(_v8, _tmp13b, _mm256_add_ps(_r5, _tmp13a));

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
            }
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
                __m256 _tmp13a = _mm256_sub_ps(_r1, _r2);
                __m256 _tmp02b = _mm256_add_ps(_r3, _r4);
                __m256 _tmp13b = _mm256_sub_ps(_r3, _r4);
                __m256 _tmp0 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_r0, _tmp02a), _tmp02b));
                __m256 _tmp1 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_v2, _tmp13b, _tmp13a));
                __m256 _tmp2 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_v4, _tmp02b, _tmp02a));
                __m256 _tmp3 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_v8, _tmp13b, _mm256_add_ps(_r5, _tmp13a)));

                if (out_elempack == 8)
                {
                    float* output0 = top_blob.channel((i + ii) / 8).row(ti * 4 + m) + (tj * 4) * 8;

                    _mm256_store_ps(output0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm256_store_ps(output0 + 8, _tmp1);
                    if (tj * 4 + 2 < outw) _mm256_store_ps(output0 + 16, _tmp2);
                    if (tj * 4 + 3 < outw) _mm256_store_ps(output0 + 24, _tmp3);
                }
                if (out_elempack == 4)
                {
                    float* output0 = top_blob.channel((i + ii) / 4).row(ti * 4 + m) + (tj * 4) * 4;
                    float* output1 = top_blob.channel((i + ii) / 4 + 1).row(ti * 4 + m) + (tj * 4) * 4;

                    _mm_store_ps(output0, _mm256_extractf128_ps(_tmp0, 0));
                    _mm_store_ps(output1, _mm256_extractf128_ps(_tmp0, 1));
                    if (tj * 4 + 1 < outw)
                    {
                        _mm_store_ps(output0 + 4, _mm256_extractf128_ps(_tmp1, 0));
                        _mm_store_ps(output1 + 4, _mm256_extractf128_ps(_tmp1, 1));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        _mm_store_ps(output0 + 8, _mm256_extractf128_ps(_tmp2, 0));
                        _mm_store_ps(output1 + 8, _mm256_extractf128_ps(_tmp2, 1));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        _mm_store_ps(output0 + 12, _mm256_extractf128_ps(_tmp3, 0));
                        _mm_store_ps(output1 + 12, _mm256_extractf128_ps(_tmp3, 1));
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

                    float* output0 = top_blob.channel(i + ii).row(ti * 4 + m) + (tj * 4);
                    float* output1 = top_blob.channel(i + ii + 1).row(ti * 4 + m) + (tj * 4);
                    float* output2 = top_blob.channel(i + ii + 2).row(ti * 4 + m) + (tj * 4);
                    float* output3 = top_blob.channel(i + ii + 3).row(ti * 4 + m) + (tj * 4);
                    float* output4 = top_blob.channel(i + ii + 4).row(ti * 4 + m) + (tj * 4);
                    float* output5 = top_blob.channel(i + ii + 5).row(ti * 4 + m) + (tj * 4);
                    float* output6 = top_blob.channel(i + ii + 6).row(ti * 4 + m) + (tj * 4);
                    float* output7 = top_blob.channel(i + ii + 7).row(ti * 4 + m) + (tj * 4);

                    output0[0] = tmp0[0];
                    output1[0] = tmp0[1];
                    output2[0] = tmp0[2];
                    output3[0] = tmp0[3];
                    output4[0] = tmp0[4];
                    output5[0] = tmp0[5];
                    output6[0] = tmp0[6];
                    output7[0] = tmp0[7];
                    if (tj * 4 + 1 < outw)
                    {
                        output0[1] = tmp1[0];
                        output1[1] = tmp1[1];
                        output2[1] = tmp1[2];
                        output3[1] = tmp1[3];
                        output4[1] = tmp1[4];
                        output5[1] = tmp1[5];
                        output6[1] = tmp1[6];
                        output7[1] = tmp1[7];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        output0[2] = tmp2[0];
                        output1[2] = tmp2[1];
                        output2[2] = tmp2[2];
                        output3[2] = tmp2[3];
                        output4[2] = tmp2[4];
                        output5[2] = tmp2[5];
                        output6[2] = tmp2[6];
                        output7[2] = tmp2[7];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        output0[3] = tmp3[0];
                        output1[3] = tmp3[1];
                        output2[3] = tmp3[2];
                        output3[3] = tmp3[3];
                        output4[3] = tmp3[4];
                        output5[3] = tmp3[5];
                        output6[3] = tmp3[6];
                        output7[3] = tmp3[7];
                    }
                }
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _bias0 = biasptr ? _mm_loadu_ps(biasptr + i + ii) : _mm_setzero_ps();

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

#ifdef _MSC_VER
            __declspec(align(16))
#else
            __attribute__((aligned(16)))
#endif
            float tmp[4][6][4];

            __m128 _v2 = _mm_set1_ps(2.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _v8 = _mm_set1_ps(8.f);

            for (int m = 0; m < 6; m++)
            {
                const float* r0 = top_tile.depth(m * 6).row(jj) + ii;
                const float* r1 = top_tile.depth(m * 6 + 1).row(jj) + ii;
                const float* r2 = top_tile.depth(m * 6 + 2).row(jj) + ii;
                const float* r3 = top_tile.depth(m * 6 + 3).row(jj) + ii;
                const float* r4 = top_tile.depth(m * 6 + 4).row(jj) + ii;
                const float* r5 = top_tile.depth(m * 6 + 5).row(jj) + ii;

                __m128 _r0 = _mm_load_ps(r0);
                __m128 _r1 = _mm_load_ps(r1);
                __m128 _r2 = _mm_load_ps(r2);
                __m128 _r3 = _mm_load_ps(r3);
                __m128 _r4 = _mm_load_ps(r4);
                __m128 _r5 = _mm_load_ps(r5);

                __m128 _tmp02a = _mm_add_ps(_r1, _r2);
                __m128 _tmp13a = _mm_sub_ps(_r1, _r2);
                __m128 _tmp02b = _mm_add_ps(_r3, _r4);
                __m128 _tmp13b = _mm_sub_ps(_r3, _r4);
                __m128 _tmp0 = _mm_add_ps(_mm_add_ps(_r0, _tmp02a), _tmp02b);
                __m128 _tmp1 = _mm_comp_fmadd_ps(_v2, _tmp13b, _tmp13a);
                __m128 _tmp2 = _mm_comp_fmadd_ps(_v4, _tmp02b, _tmp02a);
                __m128 _tmp3 = _mm_comp_fmadd_ps(_v8, _tmp13b, _mm_add_ps(_r5, _tmp13a));

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
            }
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
                __m128 _tmp13a = _mm_sub_ps(_r1, _r2);
                __m128 _tmp02b = _mm_add_ps(_r3, _r4);
                __m128 _tmp13b = _mm_sub_ps(_r3, _r4);
                __m128 _tmp0 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_r0, _tmp02a), _tmp02b));
                __m128 _tmp1 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v2, _tmp13b, _tmp13a));
                __m128 _tmp2 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v4, _tmp02b, _tmp02a));
                __m128 _tmp3 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v8, _tmp13b, _mm_add_ps(_r5, _tmp13a)));

                if (out_elempack == 4)
                {
                    float* output0 = top_blob.channel((i + ii) / 4).row(ti * 4 + m) + (tj * 4) * 4;

                    _mm_store_ps(output0, _tmp0);
                    if (tj * 4 + 1 < outw) _mm_store_ps(output0 + 4, _tmp1);
                    if (tj * 4 + 2 < outw) _mm_store_ps(output0 + 8, _tmp2);
                    if (tj * 4 + 3 < outw) _mm_store_ps(output0 + 12, _tmp3);
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

                    float* output0 = top_blob.channel(i + ii).row(ti * 4 + m) + (tj * 4);
                    float* output1 = top_blob.channel(i + ii + 1).row(ti * 4 + m) + (tj * 4);
                    float* output2 = top_blob.channel(i + ii + 2).row(ti * 4 + m) + (tj * 4);
                    float* output3 = top_blob.channel(i + ii + 3).row(ti * 4 + m) + (tj * 4);

                    output0[0] = tmp0[0];
                    output1[0] = tmp0[1];
                    output2[0] = tmp0[2];
                    output3[0] = tmp0[3];
                    if (tj * 4 + 1 < outw)
                    {
                        output0[1] = tmp1[0];
                        output1[1] = tmp1[1];
                        output2[1] = tmp1[2];
                        output3[1] = tmp1[3];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        output0[2] = tmp2[0];
                        output1[2] = tmp2[1];
                        output2[2] = tmp2[2];
                        output3[2] = tmp2[3];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        output0[3] = tmp3[0];
                        output1[3] = tmp3[1];
                        output2[3] = tmp3[2];
                        output3[3] = tmp3[3];
                    }
                }
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            float tmp[4][6][2];

            for (int m = 0; m < 6; m++)
            {
                float r00 = top_tile.depth(m * 6).row(jj)[ii];
                float r01 = top_tile.depth(m * 6).row(jj)[ii + 1];
                float r10 = top_tile.depth(m * 6 + 1).row(jj)[ii];
                float r11 = top_tile.depth(m * 6 + 1).row(jj)[ii + 1];
                float r20 = top_tile.depth(m * 6 + 2).row(jj)[ii];
                float r21 = top_tile.depth(m * 6 + 2).row(jj)[ii + 1];
                float r30 = top_tile.depth(m * 6 + 3).row(jj)[ii];
                float r31 = top_tile.depth(m * 6 + 3).row(jj)[ii + 1];
                float r40 = top_tile.depth(m * 6 + 4).row(jj)[ii];
                float r41 = top_tile.depth(m * 6 + 4).row(jj)[ii + 1];
                float r50 = top_tile.depth(m * 6 + 5).row(jj)[ii];
                float r51 = top_tile.depth(m * 6 + 5).row(jj)[ii + 1];

                float tmp00 = r00 + r10 + r20 + r30 + r40;
                float tmp01 = r01 + r11 + r21 + r31 + r41;
                float tmp10 = r10 - r20 + r30 * 2.f - r40 * 2.f;
                float tmp11 = r11 - r21 + r31 * 2.f - r41 * 2.f;
                float tmp20 = r10 + r20 + r30 * 4.f + r40 * 4.f;
                float tmp21 = r11 + r21 + r31 * 4.f + r41 * 4.f;
                float tmp30 = r10 - r20 + r30 * 8.f - r40 * 8.f + r50;
                float tmp31 = r11 - r21 + r31 * 8.f - r41 * 8.f + r51;

                tmp[0][m][0] = tmp00;
                tmp[0][m][1] = tmp01;
                tmp[1][m][0] = tmp10;
                tmp[1][m][1] = tmp11;
                tmp[2][m][0] = tmp20;
                tmp[2][m][1] = tmp21;
                tmp[3][m][0] = tmp30;
                tmp[3][m][1] = tmp31;
            }
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

                float tmp00 = bias0 + r00 + r10 + r20 + r30 + r40;
                float tmp01 = bias1 + r01 + r11 + r21 + r31 + r41;
                float tmp10 = bias0 + r10 - r20 + r30 * 2.f - r40 * 2.f;
                float tmp11 = bias1 + r11 - r21 + r31 * 2.f - r41 * 2.f;
                float tmp20 = bias0 + r10 + r20 + r30 * 4.f + r40 * 4.f;
                float tmp21 = bias1 + r11 + r21 + r31 * 4.f + r41 * 4.f;
                float tmp30 = bias0 + r10 - r20 + r30 * 8.f - r40 * 8.f + r50;
                float tmp31 = bias1 + r11 - r21 + r31 * 8.f - r41 * 8.f + r51;

                // if (out_elempack == 1)
                {
                    float* output0 = top_blob.channel(i + ii).row(ti * 4 + m) + (tj * 4);
                    float* output1 = top_blob.channel(i + ii + 1).row(ti * 4 + m) + (tj * 4);

                    output0[0] = tmp00;
                    output1[0] = tmp01;
                    if (tj * 4 + 1 < outw)
                    {
                        output0[1] = tmp10;
                        output1[1] = tmp11;
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        output0[2] = tmp20;
                        output1[2] = tmp21;
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        output0[3] = tmp30;
                        output1[3] = tmp31;
                    }
                }
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            float tmp[4][6];

            for (int m = 0; m < 6; m++)
            {
                float r0 = top_tile.depth(m * 6).row(jj)[ii];
                float r1 = top_tile.depth(m * 6 + 1).row(jj)[ii];
                float r2 = top_tile.depth(m * 6 + 2).row(jj)[ii];
                float r3 = top_tile.depth(m * 6 + 3).row(jj)[ii];
                float r4 = top_tile.depth(m * 6 + 4).row(jj)[ii];
                float r5 = top_tile.depth(m * 6 + 5).row(jj)[ii];

                float tmp0 = r0 + r1 + r2 + r3 + r4;
                float tmp1 = r1 - r2 + r3 * 2.f - r4 * 2.f;
                float tmp2 = r1 + r2 + r3 * 4.f + r4 * 4.f;
                float tmp3 = r1 - r2 + r3 * 8.f - r4 * 8.f + r5;

                tmp[0][m] = tmp0;
                tmp[1][m] = tmp1;
                tmp[2][m] = tmp2;
                tmp[3][m] = tmp3;
            }
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

                float tmp0 = bias0 + r0 + r1 + r2 + r3 + r4;
                float tmp1 = bias0 + r1 - r2 + r3 * 2.f - r4 * 2.f;
                float tmp2 = bias0 + r1 + r2 + r3 * 4.f + r4 * 4.f;
                float tmp3 = bias0 + r1 - r2 + r3 * 8.f - r4 * 8.f + r5;

                // if (out_elempack == 1)
                {
                    float* output0 = top_blob.channel(i + ii).row(ti * 4 + m) + (tj * 4);

                    output0[0] = tmp0;
                    if (tj * 4 + 1 < outw) output0[1] = tmp1;
                    if (tj * 4 + 2 < outw) output0[2] = tmp2;
                    if (tj * 4 + 3 < outw) output0[3] = tmp3;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, const Option& opt)
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

    int nT = opt.num_threads;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat B_tileX(B * TILE_N * TILE_K, 1, nT, 4u, opt.blob_allocator);
    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.blob_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_N; ppj++)
    {
        const int j = ppj * TILE_N;

        Mat B_tile = B_tileX.channel(get_omp_thread_num());

        const int max_jj = std::min((N - j), TILE_N);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd43_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk);
        }
    }

    Mat tmpX;
    if (TILE_K < K)
    {
        tmpX.create(TILE_M * TILE_N, B, nT, 4u, opt.blob_allocator);
    }

    Mat top_tileX(TILE_M, TILE_N, B, nT, 4u, opt.blob_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat tmp;
        if (K > TILE_K)
            tmp = tmpX.channel(get_omp_thread_num());

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

                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, tmp, B, max_ii, max_jj, k, max_kk, k_end);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

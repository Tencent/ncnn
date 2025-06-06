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

static void convolution_im2col_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    float* pp = AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
        const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
        const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
        const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;
        const float* p8 = (const float*)A + (i + ii + 8) * A_hstep + k;
        const float* p9 = (const float*)A + (i + ii + 9) * A_hstep + k;
        const float* pa = (const float*)A + (i + ii + 10) * A_hstep + k;
        const float* pb = (const float*)A + (i + ii + 11) * A_hstep + k;
        const float* pc = (const float*)A + (i + ii + 12) * A_hstep + k;
        const float* pd = (const float*)A + (i + ii + 13) * A_hstep + k;
        const float* pe = (const float*)A + (i + ii + 14) * A_hstep + k;
        const float* pf = (const float*)A + (i + ii + 15) * A_hstep + k;

        int kk = 0;
        for (; kk + 15 < max_kk; kk += 16)
        {
            __m512 _r0 = _mm512_loadu_ps(p0);
            __m512 _r1 = _mm512_loadu_ps(p1);
            __m512 _r2 = _mm512_loadu_ps(p2);
            __m512 _r3 = _mm512_loadu_ps(p3);
            __m512 _r4 = _mm512_loadu_ps(p4);
            __m512 _r5 = _mm512_loadu_ps(p5);
            __m512 _r6 = _mm512_loadu_ps(p6);
            __m512 _r7 = _mm512_loadu_ps(p7);
            __m512 _r8 = _mm512_loadu_ps(p8);
            __m512 _r9 = _mm512_loadu_ps(p9);
            __m512 _ra = _mm512_loadu_ps(pa);
            __m512 _rb = _mm512_loadu_ps(pb);
            __m512 _rc = _mm512_loadu_ps(pc);
            __m512 _rd = _mm512_loadu_ps(pd);
            __m512 _re = _mm512_loadu_ps(pe);
            __m512 _rf = _mm512_loadu_ps(pf);
            transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
            _mm512_store_ps(pp, _r0);
            _mm512_store_ps(pp + 16, _r1);
            _mm512_store_ps(pp + 16 * 2, _r2);
            _mm512_store_ps(pp + 16 * 3, _r3);
            _mm512_store_ps(pp + 16 * 4, _r4);
            _mm512_store_ps(pp + 16 * 5, _r5);
            _mm512_store_ps(pp + 16 * 6, _r6);
            _mm512_store_ps(pp + 16 * 7, _r7);
            _mm512_store_ps(pp + 16 * 8, _r8);
            _mm512_store_ps(pp + 16 * 9, _r9);
            _mm512_store_ps(pp + 16 * 10, _ra);
            _mm512_store_ps(pp + 16 * 11, _rb);
            _mm512_store_ps(pp + 16 * 12, _rc);
            _mm512_store_ps(pp + 16 * 13, _rd);
            _mm512_store_ps(pp + 16 * 14, _re);
            _mm512_store_ps(pp + 16 * 15, _rf);
            pp += 256;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
            p4 += 16;
            p5 += 16;
            p6 += 16;
            p7 += 16;
            p8 += 16;
            p9 += 16;
            pa += 16;
            pb += 16;
            pc += 16;
            pd += 16;
            pe += 16;
            pf += 16;
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
            pp[8] = p8[0];
            pp[9] = p9[0];
            pp[10] = pa[0];
            pp[11] = pb[0];
            pp[12] = pc[0];
            pp[13] = pd[0];
            pp[14] = pe[0];
            pp[15] = pf[0];
            pp += 16;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
            p8++;
            p9++;
            pa++;
            pb++;
            pc++;
            pd++;
            pe++;
            pf++;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
        const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
        const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
        const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(p0);
            __m256 _r1 = _mm256_loadu_ps(p1);
            __m256 _r2 = _mm256_loadu_ps(p2);
            __m256 _r3 = _mm256_loadu_ps(p3);
            __m256 _r4 = _mm256_loadu_ps(p4);
            __m256 _r5 = _mm256_loadu_ps(p5);
            __m256 _r6 = _mm256_loadu_ps(p6);
            __m256 _r7 = _mm256_loadu_ps(p7);
            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            _mm256_store_ps(pp, _r0);
            _mm256_store_ps(pp + 8, _r1);
            _mm256_store_ps(pp + 8 * 2, _r2);
            _mm256_store_ps(pp + 8 * 3, _r3);
            _mm256_store_ps(pp + 8 * 4, _r4);
            _mm256_store_ps(pp + 8 * 5, _r5);
            _mm256_store_ps(pp + 8 * 6, _r6);
            _mm256_store_ps(pp + 8 * 7, _r7);
            pp += 64;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
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
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
#if __AVX__
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(p0);
            __m256 _r1 = _mm256_loadu_ps(p1);
            __m256 _r2 = _mm256_loadu_ps(p2);
            __m256 _r3 = _mm256_loadu_ps(p3);
            transpose8x4_ps(_r0, _r1, _r2, _r3);
            _mm256_store_ps(pp, _r0);
            _mm256_store_ps(pp + 8, _r1);
            _mm256_store_ps(pp + 16, _r2);
            _mm256_store_ps(pp + 24, _r3);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
#endif // __AVX__
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128 _r0 = _mm_loadu_ps(p0);
            __m128 _r1 = _mm_loadu_ps(p1);
            __m128 _r2 = _mm_loadu_ps(p2);
            __m128 _r3 = _mm_loadu_ps(p3);
            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _mm_store_ps(pp, _r0);
            _mm_store_ps(pp + 4, _r1);
            _mm_store_ps(pp + 8, _r2);
            _mm_store_ps(pp + 12, _r3);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __SSE2__
#if __AVX__
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(p0);
            __m256 _r1 = _mm256_loadu_ps(p1);
            transpose8x2_ps(_r0, _r1);
            _mm256_storeu_ps(pp, _r0);
            _mm256_storeu_ps(pp + 8, _r1);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
#endif // __AVX__
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128 _r0 = _mm_loadu_ps(p0);
            __m128 _r1 = _mm_loadu_ps(p1);
            __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
            __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
            _mm_store_ps(pp, _tmp0);
            _mm_store_ps(pp + 4, _tmp1);
            pp += 8;
            p0 += 4;
            p1 += 4;
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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __SSE2__
#if __AVX__
        for (; kk + 7 < max_kk; kk += 8)
        {
            _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
            pp += 8;
            p0 += 8;
        }
#endif // __AVX__
        for (; kk + 3 < max_kk; kk += 4)
        {
            _mm_storeu_ps(pp, _mm_loadu_ps(p0));
            pp += 4;
            p0 += 4;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void convolution_gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
                if (pC)
                {
                    _sum0 = _mm512_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                }
                else
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
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
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
            for (; kk < max_kk; kk += 1)
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
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    _mm512_store_ps(outptr0 + 16 * 4, _sum4);
                    _mm512_store_ps(outptr0 + 16 * 5, _sum5);
                    _mm512_store_ps(outptr0 + 16 * 6, _sum6);
                    _mm512_store_ps(outptr0 + 16 * 7, _sum7);
                    _mm512_store_ps(outptr0 + 16 * 8, _sum8);
                    _mm512_store_ps(outptr0 + 16 * 9, _sum9);
                    _mm512_store_ps(outptr0 + 16 * 10, _suma);
                    _mm512_store_ps(outptr0 + 16 * 11, _sumb);
                    outptr0 += 192;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp8 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);
                    _mm512_storeu_ps(outptr0 + 16 * 2, _tmp2);
                    _mm512_storeu_ps(outptr0 + 16 * 3, _tmp3);
                    _mm512_storeu_ps(outptr0 + 16 * 4, _tmp4);
                    _mm512_storeu_ps(outptr0 + 16 * 5, _tmp5);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp7);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 2, _tmp8);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 3, _tmp9);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 4, _tmpa);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 5, _tmpb);

                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp8 = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum8 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum9 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                    _suma = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                    _sumb = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + 16, _sum4);
                    _mm512_storeu_ps(outptr0 + 32, _sum8);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16, _sum5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 32, _sum9);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _sum6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 32, _suma);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 16, _sum7);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 32, _sumb);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose16x12_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    _mm256_storeu_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm_storeu_ps(outptr0 + 8, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _mm512_extractf32x4_ps(_sum0, 3));
                    _mm256_storeu_ps(outptr0 + out_hstep * 1 + 4, _mm512_extractf32x8_ps(_sum1, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _mm512_extractf32x8_ps(_sum1, 1));
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 8, _mm512_extractf32x4_ps(_sum2, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _mm512_extractf32x4_ps(_sum2, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 3 + 4, _mm512_extractf32x8_ps(_sum2, 1));

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _mm512_extractf32x8_ps(_sum3, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 4 + 8, _mm512_extractf32x4_ps(_sum3, 2));
                    _mm_storeu_ps(outptr0 + out_hstep * 5, _mm512_extractf32x4_ps(_sum3, 3));
                    _mm256_storeu_ps(outptr0 + out_hstep * 5 + 4, _mm512_extractf32x8_ps(_sum4, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _mm512_extractf32x8_ps(_sum4, 1));
                    _mm_storeu_ps(outptr0 + out_hstep * 6 + 8, _mm512_extractf32x4_ps(_sum5, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 7, _mm512_extractf32x4_ps(_sum5, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 7 + 4, _mm512_extractf32x8_ps(_sum5, 1));

                    _mm256_storeu_ps(outptr0 + out_hstep * 8, _mm512_extractf32x8_ps(_sum6, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 8 + 8, _mm512_extractf32x4_ps(_sum6, 2));
                    _mm_storeu_ps(outptr0 + out_hstep * 9, _mm512_extractf32x4_ps(_sum6, 3));
                    _mm256_storeu_ps(outptr0 + out_hstep * 9 + 4, _mm512_extractf32x8_ps(_sum7, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 10, _mm512_extractf32x8_ps(_sum7, 1));
                    _mm_storeu_ps(outptr0 + out_hstep * 10 + 8, _mm512_extractf32x4_ps(_sum8, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 11, _mm512_extractf32x4_ps(_sum8, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 11 + 4, _mm512_extractf32x8_ps(_sum8, 1));

                    _mm256_storeu_ps(outptr0 + out_hstep * 12, _mm512_extractf32x8_ps(_sum9, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 12 + 8, _mm512_extractf32x4_ps(_sum9, 2));
                    _mm_storeu_ps(outptr0 + out_hstep * 13, _mm512_extractf32x4_ps(_sum9, 3));
                    _mm256_storeu_ps(outptr0 + out_hstep * 13 + 4, _mm512_extractf32x8_ps(_suma, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 14, _mm512_extractf32x8_ps(_suma, 1));
                    _mm_storeu_ps(outptr0 + out_hstep * 14 + 8, _mm512_extractf32x4_ps(_sumb, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 15, _mm512_extractf32x4_ps(_sumb, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 15 + 4, _mm512_extractf32x8_ps(_sumb, 1));

                    outptr0 += 12;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
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
            }

            outptr += 192;
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
                if (pC)
                {
                    _sum0 = _mm512_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
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
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
                _sum4 = _mm512_load_ps(outptr + 16 * 4);
                _sum5 = _mm512_load_ps(outptr + 16 * 5);
                _sum6 = _mm512_load_ps(outptr + 16 * 6);
                _sum7 = _mm512_load_ps(outptr + 16 * 7);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
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
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    _mm512_store_ps(outptr0 + 16 * 4, _sum4);
                    _mm512_store_ps(outptr0 + 16 * 5, _sum5);
                    _mm512_store_ps(outptr0 + 16 * 6, _sum6);
                    _mm512_store_ps(outptr0 + 16 * 7, _sum7);
                    outptr0 += 128;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);
                    _mm512_storeu_ps(outptr0 + 16 * 2, _tmp2);
                    _mm512_storeu_ps(outptr0 + 16 * 3, _tmp3);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 2, _tmp6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 3, _tmp7);

                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + 16, _sum4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16, _sum5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _sum6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 16, _sum7);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose16x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _mm512_extractf32x8_ps(_sum0, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _mm512_extractf32x8_ps(_sum1, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _mm512_extractf32x8_ps(_sum1, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _mm512_extractf32x8_ps(_sum2, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _mm512_extractf32x8_ps(_sum2, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _mm512_extractf32x8_ps(_sum3, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _mm512_extractf32x8_ps(_sum3, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 8, _mm512_extractf32x8_ps(_sum4, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 9, _mm512_extractf32x8_ps(_sum4, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 10, _mm512_extractf32x8_ps(_sum5, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 11, _mm512_extractf32x8_ps(_sum5, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 12, _mm512_extractf32x8_ps(_sum6, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 13, _mm512_extractf32x8_ps(_sum6, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 14, _mm512_extractf32x8_ps(_sum7, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 15, _mm512_extractf32x8_ps(_sum7, 1));

                    outptr0 += 8;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                _mm512_store_ps(outptr + 16 * 4, _sum4);
                _mm512_store_ps(outptr + 16 * 5, _sum5);
                _mm512_store_ps(outptr + 16 * 6, _sum6);
                _mm512_store_ps(outptr + 16 * 7, _sum7);
            }

            outptr += 128;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* pA = pAT;

            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm512_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = _mm512_setzero_ps();
                    _sum1 = _mm512_setzero_ps();
                    _sum2 = _mm512_setzero_ps();
                    _sum3 = _mm512_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
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
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    outptr0 += 64;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp3);

                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    __m128 _sum0_0 = _mm512_extractf32x4_ps(_sum0, 0);
                    __m128 _sum1_0 = _mm512_extractf32x4_ps(_sum1, 0);
                    __m128 _sum2_0 = _mm512_extractf32x4_ps(_sum2, 0);
                    __m128 _sum3_0 = _mm512_extractf32x4_ps(_sum3, 0);
                    __m128 _sum0_1 = _mm512_extractf32x4_ps(_sum0, 1);
                    __m128 _sum1_1 = _mm512_extractf32x4_ps(_sum1, 1);
                    __m128 _sum2_1 = _mm512_extractf32x4_ps(_sum2, 1);
                    __m128 _sum3_1 = _mm512_extractf32x4_ps(_sum3, 1);
                    __m128 _sum0_2 = _mm512_extractf32x4_ps(_sum0, 2);
                    __m128 _sum1_2 = _mm512_extractf32x4_ps(_sum1, 2);
                    __m128 _sum2_2 = _mm512_extractf32x4_ps(_sum2, 2);
                    __m128 _sum3_2 = _mm512_extractf32x4_ps(_sum3, 2);
                    __m128 _sum0_3 = _mm512_extractf32x4_ps(_sum0, 3);
                    __m128 _sum1_3 = _mm512_extractf32x4_ps(_sum1, 3);
                    __m128 _sum2_3 = _mm512_extractf32x4_ps(_sum2, 3);
                    __m128 _sum3_3 = _mm512_extractf32x4_ps(_sum3, 3);

                    _MM_TRANSPOSE4_PS(_sum0_0, _sum1_0, _sum2_0, _sum3_0);
                    _MM_TRANSPOSE4_PS(_sum0_1, _sum1_1, _sum2_1, _sum3_1);
                    _MM_TRANSPOSE4_PS(_sum0_2, _sum1_2, _sum2_2, _sum3_2);
                    _MM_TRANSPOSE4_PS(_sum0_3, _sum1_3, _sum2_3, _sum3_3);

                    _mm_storeu_ps(outptr0, _sum0_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4, _sum0_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5, _sum1_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6, _sum2_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7, _sum3_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 8, _sum0_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 9, _sum1_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 10, _sum2_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 11, _sum3_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 12, _sum0_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 13, _sum1_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 14, _sum2_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 15, _sum3_3);

                    outptr0 += 4;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
            }

            outptr += 64;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

            __m512 _sum0;
            __m512 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm512_loadu_ps(pC);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = _mm512_setzero_ps();
                    _sum1 = _mm512_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);

                pA += 16;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16, _sum1);
                    outptr0 += 32;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp1);

                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + 4, _mm512_extractf32x4_ps(_sum1, 0));

                    _mm_store_ps(outptr0 + out_hstep * 4, _mm512_extractf32x4_ps(_sum0, 1));
                    _mm_store_ps(outptr0 + out_hstep * 4 + 4, _mm512_extractf32x4_ps(_sum1, 1));

                    _mm_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_store_ps(outptr0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_sum1, 2));

                    _mm_store_ps(outptr0 + out_hstep * 12, _mm512_extractf32x4_ps(_sum0, 3));
                    _mm_store_ps(outptr0 + out_hstep * 12 + 4, _mm512_extractf32x4_ps(_sum1, 3));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[16];
                    float sum1[16];
                    _mm512_storeu_ps(sum0, _sum0);
                    _mm512_storeu_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 8] = sum0[8];
                    outptr0[out_hstep * 9] = sum0[9];
                    outptr0[out_hstep * 10] = sum0[10];
                    outptr0[out_hstep * 11] = sum0[11];
                    outptr0[out_hstep * 12] = sum0[12];
                    outptr0[out_hstep * 13] = sum0[13];
                    outptr0[out_hstep * 14] = sum0[14];
                    outptr0[out_hstep * 15] = sum0[15];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0[out_hstep * 8 + 1] = sum1[8];
                    outptr0[out_hstep * 9 + 1] = sum1[9];
                    outptr0[out_hstep * 10 + 1] = sum1[10];
                    outptr0[out_hstep * 11 + 1] = sum1[11];
                    outptr0[out_hstep * 12 + 1] = sum1[12];
                    outptr0[out_hstep * 13 + 1] = sum1[13];
                    outptr0[out_hstep * 14 + 1] = sum1[14];
                    outptr0[out_hstep * 15 + 1] = sum1[15];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
            }

            outptr += 32;
        }
        for (; jj < max_jj; jj += 1)
        {
            const float* pA = pAT;

            __m512 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm512_loadu_ps(pC);
                }
                else
                {
                    _sum0 = _mm512_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);

                pA += 16;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    outptr0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm256_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x8_ps(_sum0, 1));
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + out_hstep * 4, _mm512_extractf32x4_ps(_sum0, 1));
                    _mm_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_store_ps(outptr0 + out_hstep * 12, _mm512_extractf32x4_ps(_sum0, 3));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[16];
                    _mm512_storeu_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 8] = sum0[8];
                    outptr0[out_hstep * 9] = sum0[9];
                    outptr0[out_hstep * 10] = sum0[10];
                    outptr0[out_hstep * 11] = sum0[11];
                    outptr0[out_hstep * 12] = sum0[12];
                    outptr0[out_hstep * 13] = sum0[13];
                    outptr0[out_hstep * 14] = sum0[14];
                    outptr0[out_hstep * 15] = sum0[15];
                    outptr0++;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
            }

            outptr += 16;
        }

        pAT += max_kk * 16;
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
                if (pC)
                {
                    _sum0 = _mm256_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                }
                else
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
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
                _sum4 = _mm256_load_ps(outptr + 8 * 4);
                _sum5 = _mm256_load_ps(outptr + 8 * 5);
                _sum6 = _mm256_load_ps(outptr + 8 * 6);
                _sum7 = _mm256_load_ps(outptr + 8 * 7);
                _sum8 = _mm256_load_ps(outptr + 8 * 8);
                _sum9 = _mm256_load_ps(outptr + 8 * 9);
                _suma = _mm256_load_ps(outptr + 8 * 10);
                _sumb = _mm256_load_ps(outptr + 8 * 11);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
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
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                    _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                    _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                    _mm256_store_ps(outptr0 + 8 * 7, _sum7);
                    _mm256_store_ps(outptr0 + 8 * 8, _sum8);
                    _mm256_store_ps(outptr0 + 8 * 9, _sum9);
                    _mm256_store_ps(outptr0 + 8 * 10, _suma);
                    _mm256_store_ps(outptr0 + 8 * 11, _sumb);
                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp4 = _mm256_permute2f128_ps(_sum8, _sum9, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp5 = _mm256_permute2f128_ps(_suma, _sumb, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp6 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp7 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp8 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp9 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmpa = _mm256_permute2f128_ps(_sum8, _sum9, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmpb = _mm256_permute2f128_ps(_suma, _sumb, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);
                    _mm256_storeu_ps(outptr0 + 8 * 2, _tmp2);
                    _mm256_storeu_ps(outptr0 + 8 * 3, _tmp3);
                    _mm256_storeu_ps(outptr0 + 8 * 4, _tmp4);
                    _mm256_storeu_ps(outptr0 + 8 * 5, _tmp5);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp7);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 2, _tmp8);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 3, _tmp9);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 4, _tmpa);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 5, _tmpb);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _sum0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _sum4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _sum5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _sum6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _sum7);

                    __m128 _sum8_0 = _mm256_extractf128_ps(_sum8, 0);
                    __m128 _sum9_0 = _mm256_extractf128_ps(_sum9, 0);
                    __m128 _suma_0 = _mm256_extractf128_ps(_suma, 0);
                    __m128 _sumb_0 = _mm256_extractf128_ps(_sumb, 0);
                    __m128 _sum8_1 = _mm256_extractf128_ps(_sum8, 1);
                    __m128 _sum9_1 = _mm256_extractf128_ps(_sum9, 1);
                    __m128 _suma_1 = _mm256_extractf128_ps(_suma, 1);
                    __m128 _sumb_1 = _mm256_extractf128_ps(_sumb, 1);

                    _MM_TRANSPOSE4_PS(_sum8_0, _sum9_0, _suma_0, _sumb_0);
                    _MM_TRANSPOSE4_PS(_sum8_1, _sum9_1, _suma_1, _sumb_1);

                    _mm_storeu_ps(outptr0 + 8, _sum8_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 8, _sum9_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 8, _suma_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 8, _sumb_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4 + 8, _sum8_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5 + 8, _sum9_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6 + 8, _suma_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7 + 8, _sumb_1);

                    outptr0 += 12;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
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
            }

            outptr += 96;
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
                if (pC)
                {
                    _sum0 = _mm256_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
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
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
                _sum4 = _mm256_load_ps(outptr + 8 * 4);
                _sum5 = _mm256_load_ps(outptr + 8 * 5);
                _sum6 = _mm256_load_ps(outptr + 8 * 6);
                _sum7 = _mm256_load_ps(outptr + 8 * 7);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
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
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                    _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                    _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                    _mm256_store_ps(outptr0 + 8 * 7, _sum7);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp4 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp5 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp6 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp7 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);
                    _mm256_storeu_ps(outptr0 + 8 * 2, _tmp2);
                    _mm256_storeu_ps(outptr0 + 8 * 3, _tmp3);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 2, _tmp6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 3, _tmp7);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _sum0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _sum4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _sum5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _sum6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _sum7);

                    outptr0 += 8;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
                _mm256_store_ps(outptr + 8 * 4, _sum4);
                _mm256_store_ps(outptr + 8 * 5, _sum5);
                _mm256_store_ps(outptr + 8 * 6, _sum6);
                _mm256_store_ps(outptr + 8 * 7, _sum7);
            }

            outptr += 64;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm256_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = _mm256_setzero_ps();
                    _sum1 = _mm256_setzero_ps();
                    _sum2 = _mm256_setzero_ps();
                    _sum3 = _mm256_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
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
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp3);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    __m128 _sum0_0 = _mm256_extractf128_ps(_sum0, 0);
                    __m128 _sum1_0 = _mm256_extractf128_ps(_sum1, 0);
                    __m128 _sum2_0 = _mm256_extractf128_ps(_sum2, 0);
                    __m128 _sum3_0 = _mm256_extractf128_ps(_sum3, 0);
                    __m128 _sum0_1 = _mm256_extractf128_ps(_sum0, 1);
                    __m128 _sum1_1 = _mm256_extractf128_ps(_sum1, 1);
                    __m128 _sum2_1 = _mm256_extractf128_ps(_sum2, 1);
                    __m128 _sum3_1 = _mm256_extractf128_ps(_sum3, 1);

                    _MM_TRANSPOSE4_PS(_sum0_0, _sum1_0, _sum2_0, _sum3_0);
                    _MM_TRANSPOSE4_PS(_sum0_1, _sum1_1, _sum2_1, _sum3_1);

                    _mm_storeu_ps(outptr0, _sum0_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4, _sum0_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5, _sum1_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6, _sum2_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7, _sum3_1);

                    outptr0 += 4;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm256_loadu_ps(pC);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = _mm256_setzero_ps();
                    _sum1 = _mm256_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8, _sum1);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    float sum1[8];
                    _mm256_storeu_ps(sum0, _sum0);
                    _mm256_storeu_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const float* pA = pAT;

            __m256 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm256_loadu_ps(pC);
                }
                else
                {
                    _sum0 = _mm256_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm256_extractf128_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + out_hstep * 4, _mm256_extractf128_ps(_sum0, 1));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    _mm256_storeu_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
                if (pC)
                {
                    _sum0 = _mm_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                }
                else
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
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
                _sum4 = _mm_load_ps(outptr + 4 * 4);
                _sum5 = _mm_load_ps(outptr + 4 * 5);
                _sum6 = _mm_load_ps(outptr + 4 * 6);
                _sum7 = _mm_load_ps(outptr + 4 * 7);
                _sum8 = _mm_load_ps(outptr + 4 * 8);
                _sum9 = _mm_load_ps(outptr + 4 * 9);
                _suma = _mm_load_ps(outptr + 4 * 10);
                _sumb = _mm_load_ps(outptr + 4 * 11);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_loadu_ps(pA);

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
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    _mm_storeu_ps(outptr0 + 4 * 2, _sum2);
                    _mm_storeu_ps(outptr0 + 4 * 3, _sum3);
                    _mm_storeu_ps(outptr0 + 4 * 4, _sum4);
                    _mm_storeu_ps(outptr0 + 4 * 5, _sum5);
                    _mm_storeu_ps(outptr0 + 4 * 6, _sum6);
                    _mm_storeu_ps(outptr0 + 4 * 7, _sum7);
                    _mm_storeu_ps(outptr0 + 4 * 8, _sum8);
                    _mm_storeu_ps(outptr0 + 4 * 9, _sum9);
                    _mm_storeu_ps(outptr0 + 4 * 10, _suma);
                    _mm_storeu_ps(outptr0 + 4 * 11, _sumb);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);
                    _MM_TRANSPOSE4_PS(_sum4, _sum5, _sum6, _sum7);
                    _MM_TRANSPOSE4_PS(_sum8, _sum9, _suma, _sumb);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm_storeu_ps(outptr0 + 4, _sum4);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 4, _sum5);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 4, _sum6);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 4, _sum7);
                    _mm_storeu_ps(outptr0 + 8, _sum8);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 8, _sum9);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 8, _suma);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 8, _sumb);
                    outptr0 += 12;
                }
            }
            else
            {
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
            }

            outptr += 48;
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
                if (pC)
                {
                    _sum0 = _mm_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
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
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
                _sum4 = _mm_load_ps(outptr + 4 * 4);
                _sum5 = _mm_load_ps(outptr + 4 * 5);
                _sum6 = _mm_load_ps(outptr + 4 * 6);
                _sum7 = _mm_load_ps(outptr + 4 * 7);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_loadu_ps(pA);

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
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    _mm_storeu_ps(outptr0 + 4 * 2, _sum2);
                    _mm_storeu_ps(outptr0 + 4 * 3, _sum3);
                    _mm_storeu_ps(outptr0 + 4 * 4, _sum4);
                    _mm_storeu_ps(outptr0 + 4 * 5, _sum5);
                    _mm_storeu_ps(outptr0 + 4 * 6, _sum6);
                    _mm_storeu_ps(outptr0 + 4 * 7, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);
                    _MM_TRANSPOSE4_PS(_sum4, _sum5, _sum6, _sum7);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm_storeu_ps(outptr0 + 4, _sum4);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 4, _sum5);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 4, _sum6);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
                _mm_store_ps(outptr + 4 * 4, _sum4);
                _mm_store_ps(outptr + 4 * 5, _sum5);
                _mm_store_ps(outptr + 4 * 6, _sum6);
                _mm_store_ps(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm_loadu_ps(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                    _sum3 = _mm_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_loadu_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    _mm_storeu_ps(outptr0 + 4 * 2, _sum2);
                    _mm_storeu_ps(outptr0 + 4 * 3, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm_loadu_ps(pC);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_loadu_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    _mm_storeu_ps(sum0, _sum0);
                    _mm_storeu_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const float* pA = pAT;

            __m128 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm_loadu_ps(pC);
                }
                else
                {
                    _sum0 = _mm_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_loadu_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    _mm_storeu_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum02;
            __m128 _sum10;
            __m128 _sum11;
            __m128 _sum12;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = _mm_set1_ps(pC[0]);
                    _sum01 = _mm_set1_ps(pC[0]);
                    _sum02 = _mm_set1_ps(pC[0]);
                    _sum10 = _mm_set1_ps(pC[1]);
                    _sum11 = _mm_set1_ps(pC[1]);
                    _sum12 = _mm_set1_ps(pC[1]);
                }
                else
                {
                    _sum00 = _mm_setzero_ps();
                    _sum01 = _mm_setzero_ps();
                    _sum02 = _mm_setzero_ps();
                    _sum10 = _mm_setzero_ps();
                    _sum11 = _mm_setzero_ps();
                    _sum12 = _mm_setzero_ps();
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                __m128 _tmp2 = _mm_loadu_ps(outptr + 8);
                __m128 _tmp3 = _mm_loadu_ps(outptr + 12);
                __m128 _tmp4 = _mm_loadu_ps(outptr + 16);
                __m128 _tmp5 = _mm_loadu_ps(outptr + 20);
                _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _sum02 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                _sum12 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_loadu_ps(pB);
                __m128 _pB1 = _mm_loadu_ps(pB + 4);
                __m128 _pB2 = _mm_loadu_ps(pB + 8);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum00 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum00);
                _sum01 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum01);
                _sum02 = _mm_comp_fmadd_ps(_pA0, _pB2, _sum02);
                __m128 _pA1 = _mm_set1_ps(pA[1]);
                _sum10 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum10);
                _sum11 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum11);
                _sum12 = _mm_comp_fmadd_ps(_pA1, _pB2, _sum12);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr0 + 8, _sum02);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum10);
                    _mm_storeu_ps(outptr0 + out_hstep + 4, _sum11);
                    _mm_storeu_ps(outptr0 + out_hstep + 8, _sum12);
                    outptr0 += 12;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum00, _sum10);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum00, _sum10);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum01, _sum11);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum01, _sum11);
                __m128 _tmp4 = _mm_unpacklo_ps(_sum02, _sum12);
                __m128 _tmp5 = _mm_unpackhi_ps(_sum02, _sum12);
                _mm_store_ps(outptr, _tmp0);
                _mm_store_ps(outptr + 4, _tmp1);
                _mm_store_ps(outptr + 8, _tmp2);
                _mm_store_ps(outptr + 12, _tmp3);
                _mm_store_ps(outptr + 16, _tmp4);
                _mm_store_ps(outptr + 20, _tmp5);
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum10;
            __m128 _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = _mm_set1_ps(pC[0]);
                    _sum01 = _mm_set1_ps(pC[0]);
                    _sum10 = _mm_set1_ps(pC[1]);
                    _sum11 = _mm_set1_ps(pC[1]);
                }
                else
                {
                    _sum00 = _mm_setzero_ps();
                    _sum01 = _mm_setzero_ps();
                    _sum10 = _mm_setzero_ps();
                    _sum11 = _mm_setzero_ps();
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                __m128 _tmp2 = _mm_loadu_ps(outptr + 8);
                __m128 _tmp3 = _mm_loadu_ps(outptr + 12);
                _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_loadu_ps(pB);
                __m128 _pB1 = _mm_loadu_ps(pB + 4);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum00 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum00);
                _sum01 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum01);
                __m128 _pA1 = _mm_set1_ps(pA[1]);
                _sum10 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum10);
                _sum11 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum11);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum10);
                    _mm_storeu_ps(outptr0 + out_hstep + 4, _sum11);
                    outptr0 += 8;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum00, _sum10);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum00, _sum10);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum01, _sum11);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum01, _sum11);
                _mm_store_ps(outptr, _tmp0);
                _mm_store_ps(outptr + 4, _tmp1);
                _mm_store_ps(outptr + 8, _tmp2);
                _mm_store_ps(outptr + 12, _tmp3);
            }

            outptr += 16;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm_set1_ps(pC[0]);
                    _sum1 = _mm_set1_ps(pC[1]);
                }
                else
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                _sum0 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum1 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB = _mm_loadu_ps(pB);

                _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[0]), _pB, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[1]), _pB, _sum1);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum1);
                    outptr0 += 4;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum0, _sum1);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum0, _sum1);
                _mm_storeu_ps(outptr, _tmp0);
                _mm_storeu_ps(outptr + 4, _tmp1);
            }

            outptr += 8;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum10 = pC[0];
                    sum11 = pC[1];
                }
                else
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
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
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum10;
                    outptr0[out_hstep] = sum01;
                    outptr0[out_hstep + 1] = sum11;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm_set1_ps(pC[0]);
                    _sum1 = _mm_set1_ps(pC[0]);
                    _sum2 = _mm_set1_ps(pC[0]);
                }
                else
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm_loadu_ps(outptr);
                _sum1 = _mm_loadu_ps(outptr + 4);
                _sum2 = _mm_loadu_ps(outptr + 8);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_loadu_ps(pB);
                __m128 _pB1 = _mm_loadu_ps(pB + 4);
                __m128 _pB2 = _mm_loadu_ps(pB + 8);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA0, _pB2, _sum2);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    _mm_storeu_ps(outptr0 + 8, _sum2);
                    outptr0 += 12;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
                _mm_storeu_ps(outptr + 8, _sum2);
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm_set1_ps(pC[0]);
                    _sum1 = _mm_set1_ps(pC[0]);
                }
                else
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                }
            }
            else
            {
                _sum0 = _mm_loadu_ps(outptr);
                _sum1 = _mm_loadu_ps(outptr + 4);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_loadu_ps(pB);
                __m128 _pB1 = _mm_loadu_ps(pB + 4);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
            }

            outptr += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum;

            if (k == 0)
            {
                if (pC)
                {
                    _sum = _mm_set1_ps(pC[0]);
                }
                else
                {
                    _sum = _mm_setzero_ps();
                }
            }
            else
            {
                _sum = _mm_loadu_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB = _mm_loadu_ps(pB);

                _sum = _mm_comp_fmadd_ps(_mm_set1_ps(pA[0]), _pB, _sum);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum);
            }

            outptr += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                if (pC)
                {
                    sum = pC[0];
                }
                else
                {
                    sum = 0.f;
                }
            }
            else
            {
                sum = outptr[0];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp32 = (int)(get_cpu_level2_cache_size() / sizeof(float));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __AVX512F__
        int tile_size = (l2_cache_size_fp32 - 64) / 16;
#elif __AVX__
        int tile_size = (l2_cache_size_fp32 - 32) / 8;
#elif __SSE2__
        int tile_size = (l2_cache_size_fp32 - 16) / 8;
#else
        int tile_size = (l2_cache_size_fp32 - 2) / 3;
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
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

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

#if __AVX512F__
        TILE_N = std::max(4, TILE_N);
#elif __AVX__
        TILE_N = std::max(4, TILE_N);
#elif __SSE2__
        TILE_N = std::max(4, TILE_N);
#else
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    float* pp = B;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
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
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                _mm512_store_ps(pp + 16 * 4, _r4);
                _mm512_store_ps(pp + 16 * 5, _r5);
                _mm512_store_ps(pp + 16 * 6, _r6);
                _mm512_store_ps(pp + 16 * 7, _r7);
                _mm512_store_ps(pp + 16 * 8, _r8);
                _mm512_store_ps(pp + 16 * 9, _r9);
                _mm512_store_ps(pp + 16 * 10, _ra);
                _mm512_store_ps(pp + 16 * 11, _rb);
                pp += 192;
                p0 += bottom_blob.cstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
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
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8 * 1, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                _mm256_store_ps(pp + 8 * 4, _r4);
                _mm256_store_ps(pp + 8 * 5, _r5);
                _mm256_store_ps(pp + 8 * 6, _r6);
                _mm256_store_ps(pp + 8 * 7, _r7);
                _mm256_store_ps(pp + 8 * 8, _r8);
                _mm256_store_ps(pp + 8 * 9, _r9);
                _mm256_store_ps(pp + 8 * 10, _ra);
                _mm256_store_ps(pp + 8 * 11, _rb);
                pp += 96;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
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
                _mm_store_ps(pp + 4 * 1, _r4);
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
                pp += 48;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p0 + 4);
                __m128 _r2 = _mm_loadu_ps(p0 + 8);
                _mm_storeu_ps(pp, _r0);
                _mm_storeu_ps(pp + 4, _r1);
                _mm_storeu_ps(pp + 8, _r2);
                pp += 12;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
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
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                _mm512_store_ps(pp + 16 * 4, _r4);
                _mm512_store_ps(pp + 16 * 5, _r5);
                _mm512_store_ps(pp + 16 * 6, _r6);
                _mm512_store_ps(pp + 16 * 7, _r7);
                pp += 128;
                p0 += bottom_blob.cstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
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
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8 * 1, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                _mm256_store_ps(pp + 8 * 4, _r4);
                _mm256_store_ps(pp + 8 * 5, _r5);
                _mm256_store_ps(pp + 8 * 6, _r6);
                _mm256_store_ps(pp + 8 * 7, _r7);
                pp += 64;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
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
                _mm_store_ps(pp + 4 * 1, _r4);
                _mm_store_ps(pp + 4 * 2, _r1);
                _mm_store_ps(pp + 4 * 3, _r5);
                _mm_store_ps(pp + 4 * 4, _r2);
                _mm_store_ps(pp + 4 * 5, _r6);
                _mm_store_ps(pp + 4 * 6, _r3);
                _mm_store_ps(pp + 4 * 7, _r7);
                pp += 32;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p0 + 4);
                _mm_storeu_ps(pp, _r0);
                _mm_storeu_ps(pp + 4, _r1);
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                pp += 64;
                p0 += bottom_blob.cstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8 * 1, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                pp += 32;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4 * 1, _r1);
                _mm_store_ps(pp + 4 * 2, _r2);
                _mm_store_ps(pp + 4 * 3, _r3);
                pp += 16;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_ps(pp, _mm_loadu_ps(p0));
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16);
                transpose16x2_ps(_r0, _r1);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16, _r1);
                pp += 32;
                p0 += bottom_blob.cstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8);
                transpose8x2_ps(_r0, _r1);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8, _r1);
                pp += 16;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x2
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4);
                __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                _mm_store_ps(pp, _tmp0);
                _mm_store_ps(pp + 4, _tmp1);
                pp += 8;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
            {
                _mm512_store_ps(pp, _mm512_load_ps(p0));
                pp += 16;
                p0 += bottom_blob.cstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                _mm256_store_ps(pp, _mm256_load_ps(p0));
                pp += 8;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
                pp += 4;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += bottom_blob.cstep;
            }
        }
    }
}

static inline void convolution_im2col_input_tile_impl(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    float* pp = B;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
    for (; jj + 11 < max_jj; jj += 12)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dy8 = (j + jj + 8) / outw;
        int dy9 = (j + jj + 9) / outw;
        int dya = (j + jj + 10) / outw;
        int dyb = (j + jj + 11) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;
        int dx8 = (j + jj + 8) % outw;
        int dx9 = (j + jj + 9) % outw;
        int dxa = (j + jj + 10) % outw;
        int dxb = (j + jj + 11) % outw;

        if (dy0 == dyb)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = _mm512_load_ps(sptr);
                    __m512 _r1 = _mm512_load_ps(sptr + stride_w * 16);
                    __m512 _r2 = _mm512_load_ps(sptr + stride_w * 32);
                    __m512 _r3 = _mm512_load_ps(sptr + stride_w * 48);
                    __m512 _r4 = _mm512_load_ps(sptr + stride_w * 64);
                    __m512 _r5 = _mm512_load_ps(sptr + stride_w * 80);
                    __m512 _r6 = _mm512_load_ps(sptr + stride_w * 96);
                    __m512 _r7 = _mm512_load_ps(sptr + stride_w * 112);
                    __m512 _r8 = _mm512_load_ps(sptr + stride_w * 128);
                    __m512 _r9 = _mm512_load_ps(sptr + stride_w * 144);
                    __m512 _ra = _mm512_load_ps(sptr + stride_w * 160);
                    __m512 _rb = _mm512_load_ps(sptr + stride_w * 176);
                    transpose16x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                    _mm512_store_ps(pp, _r0);
                    _mm512_store_ps(pp + 16 * 1, _r1);
                    _mm512_store_ps(pp + 16 * 2, _r2);
                    _mm512_store_ps(pp + 16 * 3, _r3);
                    _mm512_store_ps(pp + 16 * 4, _r4);
                    _mm512_store_ps(pp + 16 * 5, _r5);
                    _mm512_store_ps(pp + 16 * 6, _r6);
                    _mm512_store_ps(pp + 16 * 7, _r7);
                    _mm512_store_ps(pp + 16 * 8, _r8);
                    _mm512_store_ps(pp + 16 * 9, _r9);
                    _mm512_store_ps(pp + 16 * 10, _ra);
                    _mm512_store_ps(pp + 16 * 11, _rb);
                    pp += 192;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = _mm256_load_ps(sptr);
                    __m256 _r1 = _mm256_load_ps(sptr + stride_w * 8);
                    __m256 _r2 = _mm256_load_ps(sptr + stride_w * 16);
                    __m256 _r3 = _mm256_load_ps(sptr + stride_w * 24);
                    __m256 _r4 = _mm256_load_ps(sptr + stride_w * 32);
                    __m256 _r5 = _mm256_load_ps(sptr + stride_w * 40);
                    __m256 _r6 = _mm256_load_ps(sptr + stride_w * 48);
                    __m256 _r7 = _mm256_load_ps(sptr + stride_w * 56);
                    __m256 _r8 = _mm256_load_ps(sptr + stride_w * 64);
                    __m256 _r9 = _mm256_load_ps(sptr + stride_w * 72);
                    __m256 _ra = _mm256_load_ps(sptr + stride_w * 80);
                    __m256 _rb = _mm256_load_ps(sptr + stride_w * 88);
                    transpose8x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                    _mm256_store_ps(pp, _r0);
                    _mm256_store_ps(pp + 8 * 1, _r1);
                    _mm256_store_ps(pp + 8 * 2, _r2);
                    _mm256_store_ps(pp + 8 * 3, _r3);
                    _mm256_store_ps(pp + 8 * 4, _r4);
                    _mm256_store_ps(pp + 8 * 5, _r5);
                    _mm256_store_ps(pp + 8 * 6, _r6);
                    _mm256_store_ps(pp + 8 * 7, _r7);
                    _mm256_store_ps(pp + 8 * 8, _r8);
                    _mm256_store_ps(pp + 8 * 9, _r9);
                    _mm256_store_ps(pp + 8 * 10, _ra);
                    _mm256_store_ps(pp + 8 * 11, _rb);
                    pp += 96;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = _mm_load_ps(sptr);
                    __m128 _r1 = _mm_load_ps(sptr + stride_w * 4);
                    __m128 _r2 = _mm_load_ps(sptr + stride_w * 8);
                    __m128 _r3 = _mm_load_ps(sptr + stride_w * 12);
                    __m128 _r4 = _mm_load_ps(sptr + stride_w * 16);
                    __m128 _r5 = _mm_load_ps(sptr + stride_w * 20);
                    __m128 _r6 = _mm_load_ps(sptr + stride_w * 24);
                    __m128 _r7 = _mm_load_ps(sptr + stride_w * 28);
                    __m128 _r8 = _mm_load_ps(sptr + stride_w * 32);
                    __m128 _r9 = _mm_load_ps(sptr + stride_w * 36);
                    __m128 _ra = _mm_load_ps(sptr + stride_w * 40);
                    __m128 _rb = _mm_load_ps(sptr + stride_w * 44);
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                    _mm_store_ps(pp, _r0);
                    _mm_store_ps(pp + 4 * 1, _r4);
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
                    pp += 48;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp[8] = sptr[stride_w * 8];
                    pp[9] = sptr[stride_w * 9];
                    pp[10] = sptr[stride_w * 10];
                    pp[11] = sptr[stride_w * 11];
                    pp += 12;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int x8 = stride_w * dx8 + dilation_w * v;
                int x9 = stride_w * dx9 + dilation_w * v;
                int xa = stride_w * dxa + dilation_w * v;
                int xb = stride_w * dxb + dilation_w * v;

                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;
                int y8 = stride_h * dy8 + dilation_h * u;
                int y9 = stride_h * dy9 + dilation_h * u;
                int ya = stride_h * dya + dilation_h * u;
                int yb = stride_h * dyb + dilation_h * u;

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;
                const float* sptr2 = img.row(y2) + x2 * elempack;
                const float* sptr3 = img.row(y3) + x3 * elempack;
                const float* sptr4 = img.row(y4) + x4 * elempack;
                const float* sptr5 = img.row(y5) + x5 * elempack;
                const float* sptr6 = img.row(y6) + x6 * elempack;
                const float* sptr7 = img.row(y7) + x7 * elempack;
                const float* sptr8 = img.row(y8) + x8 * elempack;
                const float* sptr9 = img.row(y9) + x9 * elempack;
                const float* sptra = img.row(ya) + xa * elempack;
                const float* sptrb = img.row(yb) + xb * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = _mm512_load_ps(sptr0);
                    __m512 _r1 = _mm512_load_ps(sptr1);
                    __m512 _r2 = _mm512_load_ps(sptr2);
                    __m512 _r3 = _mm512_load_ps(sptr3);
                    __m512 _r4 = _mm512_load_ps(sptr4);
                    __m512 _r5 = _mm512_load_ps(sptr5);
                    __m512 _r6 = _mm512_load_ps(sptr6);
                    __m512 _r7 = _mm512_load_ps(sptr7);
                    __m512 _r8 = _mm512_load_ps(sptr8);
                    __m512 _r9 = _mm512_load_ps(sptr9);
                    __m512 _ra = _mm512_load_ps(sptra);
                    __m512 _rb = _mm512_load_ps(sptrb);
                    transpose16x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                    _mm512_store_ps(pp, _r0);
                    _mm512_store_ps(pp + 16 * 1, _r1);
                    _mm512_store_ps(pp + 16 * 2, _r2);
                    _mm512_store_ps(pp + 16 * 3, _r3);
                    _mm512_store_ps(pp + 16 * 4, _r4);
                    _mm512_store_ps(pp + 16 * 5, _r5);
                    _mm512_store_ps(pp + 16 * 6, _r6);
                    _mm512_store_ps(pp + 16 * 7, _r7);
                    _mm512_store_ps(pp + 16 * 8, _r8);
                    _mm512_store_ps(pp + 16 * 9, _r9);
                    _mm512_store_ps(pp + 16 * 10, _ra);
                    _mm512_store_ps(pp + 16 * 11, _rb);
                    pp += 192;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = _mm256_load_ps(sptr0);
                    __m256 _r1 = _mm256_load_ps(sptr1);
                    __m256 _r2 = _mm256_load_ps(sptr2);
                    __m256 _r3 = _mm256_load_ps(sptr3);
                    __m256 _r4 = _mm256_load_ps(sptr4);
                    __m256 _r5 = _mm256_load_ps(sptr5);
                    __m256 _r6 = _mm256_load_ps(sptr6);
                    __m256 _r7 = _mm256_load_ps(sptr7);
                    __m256 _r8 = _mm256_load_ps(sptr8);
                    __m256 _r9 = _mm256_load_ps(sptr9);
                    __m256 _ra = _mm256_load_ps(sptra);
                    __m256 _rb = _mm256_load_ps(sptrb);
                    transpose8x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                    _mm256_store_ps(pp, _r0);
                    _mm256_store_ps(pp + 8 * 1, _r1);
                    _mm256_store_ps(pp + 8 * 2, _r2);
                    _mm256_store_ps(pp + 8 * 3, _r3);
                    _mm256_store_ps(pp + 8 * 4, _r4);
                    _mm256_store_ps(pp + 8 * 5, _r5);
                    _mm256_store_ps(pp + 8 * 6, _r6);
                    _mm256_store_ps(pp + 8 * 7, _r7);
                    _mm256_store_ps(pp + 8 * 8, _r8);
                    _mm256_store_ps(pp + 8 * 9, _r9);
                    _mm256_store_ps(pp + 8 * 10, _ra);
                    _mm256_store_ps(pp + 8 * 11, _rb);
                    pp += 96;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = _mm_load_ps(sptr0);
                    __m128 _r1 = _mm_load_ps(sptr1);
                    __m128 _r2 = _mm_load_ps(sptr2);
                    __m128 _r3 = _mm_load_ps(sptr3);
                    __m128 _r4 = _mm_load_ps(sptr4);
                    __m128 _r5 = _mm_load_ps(sptr5);
                    __m128 _r6 = _mm_load_ps(sptr6);
                    __m128 _r7 = _mm_load_ps(sptr7);
                    __m128 _r8 = _mm_load_ps(sptr8);
                    __m128 _r9 = _mm_load_ps(sptr9);
                    __m128 _ra = _mm_load_ps(sptra);
                    __m128 _rb = _mm_load_ps(sptrb);
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                    _mm_store_ps(pp, _r0);
                    _mm_store_ps(pp + 4 * 1, _r4);
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
                    pp += 48;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp[8] = sptr8[0];
                    pp[9] = sptr9[0];
                    pp[10] = sptra[0];
                    pp[11] = sptrb[0];
                    pp += 12;
                }
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;

        if (dy0 == dy7)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = _mm512_load_ps(sptr);
                    __m512 _r1 = _mm512_load_ps(sptr + stride_w * 16);
                    __m512 _r2 = _mm512_load_ps(sptr + stride_w * 32);
                    __m512 _r3 = _mm512_load_ps(sptr + stride_w * 48);
                    __m512 _r4 = _mm512_load_ps(sptr + stride_w * 64);
                    __m512 _r5 = _mm512_load_ps(sptr + stride_w * 80);
                    __m512 _r6 = _mm512_load_ps(sptr + stride_w * 96);
                    __m512 _r7 = _mm512_load_ps(sptr + stride_w * 112);
                    transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    _mm512_store_ps(pp, _r0);
                    _mm512_store_ps(pp + 16 * 1, _r1);
                    _mm512_store_ps(pp + 16 * 2, _r2);
                    _mm512_store_ps(pp + 16 * 3, _r3);
                    _mm512_store_ps(pp + 16 * 4, _r4);
                    _mm512_store_ps(pp + 16 * 5, _r5);
                    _mm512_store_ps(pp + 16 * 6, _r6);
                    _mm512_store_ps(pp + 16 * 7, _r7);
                    pp += 128;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = _mm256_load_ps(sptr);
                    __m256 _r1 = _mm256_load_ps(sptr + stride_w * 8);
                    __m256 _r2 = _mm256_load_ps(sptr + stride_w * 16);
                    __m256 _r3 = _mm256_load_ps(sptr + stride_w * 24);
                    __m256 _r4 = _mm256_load_ps(sptr + stride_w * 32);
                    __m256 _r5 = _mm256_load_ps(sptr + stride_w * 40);
                    __m256 _r6 = _mm256_load_ps(sptr + stride_w * 48);
                    __m256 _r7 = _mm256_load_ps(sptr + stride_w * 56);
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    _mm256_store_ps(pp, _r0);
                    _mm256_store_ps(pp + 8 * 1, _r1);
                    _mm256_store_ps(pp + 8 * 2, _r2);
                    _mm256_store_ps(pp + 8 * 3, _r3);
                    _mm256_store_ps(pp + 8 * 4, _r4);
                    _mm256_store_ps(pp + 8 * 5, _r5);
                    _mm256_store_ps(pp + 8 * 6, _r6);
                    _mm256_store_ps(pp + 8 * 7, _r7);
                    pp += 64;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = _mm_load_ps(sptr);
                    __m128 _r1 = _mm_load_ps(sptr + stride_w * 4);
                    __m128 _r2 = _mm_load_ps(sptr + stride_w * 8);
                    __m128 _r3 = _mm_load_ps(sptr + stride_w * 12);
                    __m128 _r4 = _mm_load_ps(sptr + stride_w * 16);
                    __m128 _r5 = _mm_load_ps(sptr + stride_w * 20);
                    __m128 _r6 = _mm_load_ps(sptr + stride_w * 24);
                    __m128 _r7 = _mm_load_ps(sptr + stride_w * 28);
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _mm_store_ps(pp, _r0);
                    _mm_store_ps(pp + 4 * 1, _r4);
                    _mm_store_ps(pp + 4 * 2, _r1);
                    _mm_store_ps(pp + 4 * 3, _r5);
                    _mm_store_ps(pp + 4 * 4, _r2);
                    _mm_store_ps(pp + 4 * 5, _r6);
                    _mm_store_ps(pp + 4 * 6, _r3);
                    _mm_store_ps(pp + 4 * 7, _r7);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp += 8;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;
                const float* sptr2 = img.row(y2) + x2 * elempack;
                const float* sptr3 = img.row(y3) + x3 * elempack;
                const float* sptr4 = img.row(y4) + x4 * elempack;
                const float* sptr5 = img.row(y5) + x5 * elempack;
                const float* sptr6 = img.row(y6) + x6 * elempack;
                const float* sptr7 = img.row(y7) + x7 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = _mm512_load_ps(sptr0);
                    __m512 _r1 = _mm512_load_ps(sptr1);
                    __m512 _r2 = _mm512_load_ps(sptr2);
                    __m512 _r3 = _mm512_load_ps(sptr3);
                    __m512 _r4 = _mm512_load_ps(sptr4);
                    __m512 _r5 = _mm512_load_ps(sptr5);
                    __m512 _r6 = _mm512_load_ps(sptr6);
                    __m512 _r7 = _mm512_load_ps(sptr7);
                    transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    _mm512_store_ps(pp, _r0);
                    _mm512_store_ps(pp + 16 * 1, _r1);
                    _mm512_store_ps(pp + 16 * 2, _r2);
                    _mm512_store_ps(pp + 16 * 3, _r3);
                    _mm512_store_ps(pp + 16 * 4, _r4);
                    _mm512_store_ps(pp + 16 * 5, _r5);
                    _mm512_store_ps(pp + 16 * 6, _r6);
                    _mm512_store_ps(pp + 16 * 7, _r7);
                    pp += 128;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = _mm256_load_ps(sptr0);
                    __m256 _r1 = _mm256_load_ps(sptr1);
                    __m256 _r2 = _mm256_load_ps(sptr2);
                    __m256 _r3 = _mm256_load_ps(sptr3);
                    __m256 _r4 = _mm256_load_ps(sptr4);
                    __m256 _r5 = _mm256_load_ps(sptr5);
                    __m256 _r6 = _mm256_load_ps(sptr6);
                    __m256 _r7 = _mm256_load_ps(sptr7);
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    _mm256_store_ps(pp, _r0);
                    _mm256_store_ps(pp + 8 * 1, _r1);
                    _mm256_store_ps(pp + 8 * 2, _r2);
                    _mm256_store_ps(pp + 8 * 3, _r3);
                    _mm256_store_ps(pp + 8 * 4, _r4);
                    _mm256_store_ps(pp + 8 * 5, _r5);
                    _mm256_store_ps(pp + 8 * 6, _r6);
                    _mm256_store_ps(pp + 8 * 7, _r7);
                    pp += 64;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = _mm_load_ps(sptr0);
                    __m128 _r1 = _mm_load_ps(sptr1);
                    __m128 _r2 = _mm_load_ps(sptr2);
                    __m128 _r3 = _mm_load_ps(sptr3);
                    __m128 _r4 = _mm_load_ps(sptr4);
                    __m128 _r5 = _mm_load_ps(sptr5);
                    __m128 _r6 = _mm_load_ps(sptr6);
                    __m128 _r7 = _mm_load_ps(sptr7);
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _mm_store_ps(pp, _r0);
                    _mm_store_ps(pp + 4 * 1, _r4);
                    _mm_store_ps(pp + 4 * 2, _r1);
                    _mm_store_ps(pp + 4 * 3, _r5);
                    _mm_store_ps(pp + 4 * 4, _r2);
                    _mm_store_ps(pp + 4 * 5, _r6);
                    _mm_store_ps(pp + 4 * 6, _r3);
                    _mm_store_ps(pp + 4 * 7, _r7);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp += 8;
                }
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;

        if (dy0 == dy3)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = _mm512_load_ps(sptr);
                    __m512 _r1 = _mm512_load_ps(sptr + stride_w * 16);
                    __m512 _r2 = _mm512_load_ps(sptr + stride_w * 32);
                    __m512 _r3 = _mm512_load_ps(sptr + stride_w * 48);
                    transpose16x4_ps(_r0, _r1, _r2, _r3);
                    _mm512_store_ps(pp, _r0);
                    _mm512_store_ps(pp + 16 * 1, _r1);
                    _mm512_store_ps(pp + 16 * 2, _r2);
                    _mm512_store_ps(pp + 16 * 3, _r3);
                    pp += 64;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = _mm256_load_ps(sptr);
                    __m256 _r1 = _mm256_load_ps(sptr + stride_w * 8);
                    __m256 _r2 = _mm256_load_ps(sptr + stride_w * 16);
                    __m256 _r3 = _mm256_load_ps(sptr + stride_w * 24);
                    transpose8x4_ps(_r0, _r1, _r2, _r3);
                    _mm256_store_ps(pp, _r0);
                    _mm256_store_ps(pp + 8 * 1, _r1);
                    _mm256_store_ps(pp + 8 * 2, _r2);
                    _mm256_store_ps(pp + 8 * 3, _r3);
                    pp += 32;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = _mm_load_ps(sptr);
                    __m128 _r1 = _mm_load_ps(sptr + stride_w * 4);
                    __m128 _r2 = _mm_load_ps(sptr + stride_w * 8);
                    __m128 _r3 = _mm_load_ps(sptr + stride_w * 12);
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _mm_store_ps(pp, _r0);
                    _mm_store_ps(pp + 4 * 1, _r1);
                    _mm_store_ps(pp + 4 * 2, _r2);
                    _mm_store_ps(pp + 4 * 3, _r3);
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp += 4;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;
                const float* sptr2 = img.row(y2) + x2 * elempack;
                const float* sptr3 = img.row(y3) + x3 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = _mm512_load_ps(sptr0);
                    __m512 _r1 = _mm512_load_ps(sptr1);
                    __m512 _r2 = _mm512_load_ps(sptr2);
                    __m512 _r3 = _mm512_load_ps(sptr3);
                    transpose16x4_ps(_r0, _r1, _r2, _r3);
                    _mm512_store_ps(pp, _r0);
                    _mm512_store_ps(pp + 16 * 1, _r1);
                    _mm512_store_ps(pp + 16 * 2, _r2);
                    _mm512_store_ps(pp + 16 * 3, _r3);
                    pp += 64;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = _mm256_load_ps(sptr0);
                    __m256 _r1 = _mm256_load_ps(sptr1);
                    __m256 _r2 = _mm256_load_ps(sptr2);
                    __m256 _r3 = _mm256_load_ps(sptr3);
                    transpose8x4_ps(_r0, _r1, _r2, _r3);
                    _mm256_store_ps(pp, _r0);
                    _mm256_store_ps(pp + 8 * 1, _r1);
                    _mm256_store_ps(pp + 8 * 2, _r2);
                    _mm256_store_ps(pp + 8 * 3, _r3);
                    pp += 32;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = _mm_load_ps(sptr0);
                    __m128 _r1 = _mm_load_ps(sptr1);
                    __m128 _r2 = _mm_load_ps(sptr2);
                    __m128 _r3 = _mm_load_ps(sptr3);
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _mm_store_ps(pp, _r0);
                    _mm_store_ps(pp + 4 * 1, _r1);
                    _mm_store_ps(pp + 4 * 2, _r2);
                    _mm_store_ps(pp + 4 * 3, _r3);
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp += 4;
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

        if (dy0 == dy1)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = _mm512_load_ps(sptr);
                    __m512 _r1 = _mm512_load_ps(sptr + stride_w * 16);
                    transpose16x2_ps(_r0, _r1);
                    _mm512_store_ps(pp, _r0);
                    _mm512_store_ps(pp + 16, _r1);
                    pp += 32;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = _mm256_load_ps(sptr);
                    __m256 _r1 = _mm256_load_ps(sptr + stride_w * 8);
                    transpose8x2_ps(_r0, _r1);
                    _mm256_store_ps(pp, _r0);
                    _mm256_store_ps(pp + 8, _r1);
                    pp += 16;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = _mm_load_ps(sptr);
                    __m128 _r1 = _mm_load_ps(sptr + stride_w * 4);
                    __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                    __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                    _mm_store_ps(pp, _tmp0);
                    _mm_store_ps(pp + 4, _tmp1);
                    pp += 8;
                }
#endif // __SSE2__
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp += 2;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = _mm512_load_ps(sptr0);
                    __m512 _r1 = _mm512_load_ps(sptr1);
                    transpose16x2_ps(_r0, _r1);
                    _mm512_store_ps(pp, _r0);
                    _mm512_store_ps(pp + 16, _r1);
                    pp += 32;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = _mm256_load_ps(sptr0);
                    __m256 _r1 = _mm256_load_ps(sptr1);
                    transpose8x2_ps(_r0, _r1);
                    _mm256_store_ps(pp, _r0);
                    _mm256_store_ps(pp + 8, _r1);
                    pp += 16;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = _mm_load_ps(sptr0);
                    __m128 _r1 = _mm_load_ps(sptr1);
                    __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                    __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                    _mm_store_ps(pp, _tmp0);
                    _mm_store_ps(pp + 4, _tmp1);
                    pp += 8;
                }
#endif // __SSE2__
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp += 2;
                }
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const float* sptr = img.row(y) + x * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                _mm512_store_ps(pp, _mm512_load_ps(sptr));
                pp += 16;
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                _mm256_store_ps(pp, _mm256_load_ps(sptr));
                pp += 8;
            }
#endif // __AVX__
            if (elempack == 4)
            {
                _mm_store_ps(pp, _mm_load_ps(sptr));
                pp += 4;
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
#if __AVX512F__
void convolution_im2col_input_tile_avx512(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#elif __AVX__
void convolution_im2col_input_tile_avx(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#else
void convolution_im2col_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#endif
{
    convolution_im2col_input_tile_impl(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

#if __AVX512F__
template void convolution_im2col_input_tile_avx512<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#elif __AVX__
template void convolution_im2col_input_tile_avx<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#else
template void convolution_im2col_input_tile<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#endif

static void convolution_im2col_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    convolution_im2col_input_tile_impl(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        elempack = inch % 16 == 0 ? 16 : inch % 8 == 0 ? 8 : inch % 4 == 0 ? 4 : 1;
#elif __AVX__
        elempack = inch % 8 == 0 ? 8 : inch % 4 == 0 ? 4 : 1;
#else
        elempack = inch % 4 == 0 ? 4 : 1;
#endif
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

        A_data.create(maxk * inch, outch);

        for (int q = 0; q < outch; q += 1)
        {
            float* g00 = A_data.row(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);
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
        convolution_im2col_input_tile(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
    {
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT_tileX.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }

    return 0;
}

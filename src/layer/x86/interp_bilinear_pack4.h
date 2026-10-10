// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void resize_bilinear_image_pack4(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)4 * 4u, 4);
    Mat rowsbuf1(w, (size_t)4 * 4u, 4);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
#if __AVX__
#if __AVX512F__
            for (; dx + 3 < w; dx += 4)
            {
                int sx0 = xofs[dx] * 4;
                int sx1 = xofs[dx + 1] * 4;
                int sx2 = xofs[dx + 2] * 4;
                int sx3 = xofs[dx + 3] * 4;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2], alphap[4], alphap[4], alphap[4], alphap[4], alphap[6], alphap[6], alphap[6], alphap[6]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3], alphap[5], alphap[5], alphap[5], alphap[5], alphap[7], alphap[7], alphap[7], alphap[7]);

                __m128 _S10_0 = _mm_load_ps(S1 + sx0);
                __m128 _S10_1 = _mm_load_ps(S1 + sx1);
                __m128 _S10_2 = _mm_load_ps(S1 + sx2);
                __m128 _S10_3 = _mm_load_ps(S1 + sx3);
                __m128 _S11_0 = _mm_load_ps(S1 + sx0 + 4);
                __m128 _S11_1 = _mm_load_ps(S1 + sx1 + 4);
                __m128 _S11_2 = _mm_load_ps(S1 + sx2 + 4);
                __m128 _S11_3 = _mm_load_ps(S1 + sx3 + 4);

                __m512 _S10 = combine4x4_ps(_S10_0, _S10_1, _S10_2, _S10_3);
                __m512 _S11 = combine4x4_ps(_S11_0, _S11_1, _S11_2, _S11_3);

                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_storeu_ps(rows1p + dx * 4, _rows1);

                alphap += 8;
            }
#endif // __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 4;
                int sx1 = xofs[dx + 1] * 4;

                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3]);

                __m256 _S10 = combine4x2_ps(_mm_load_ps(S1 + sx0), _mm_load_ps(S1 + sx1));
                __m256 _S11 = combine4x2_ps(_mm_load_ps(S1 + sx0 + 4), _mm_load_ps(S1 + sx1 + 4));

                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_storeu_ps(rows1p + dx * 4, _rows1);

                alphap += 4;
            }
#endif // __AVX__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S1p = S1 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);

                __m128 _S10 = _mm_load_ps(S1p);
                __m128 _S11 = _mm_load_ps(S1p + 4);
                __m128 _rows1 = _mm_mul_ps(_S10, _a0);
                _rows1 = _mm_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm_store_ps(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src.row(sy);
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
#if __AVX__
#if __AVX512F__
            for (; dx + 3 < w; dx += 4)
            {
                int sx0 = xofs[dx] * 4;
                int sx1 = xofs[dx + 1] * 4;
                int sx2 = xofs[dx + 2] * 4;
                int sx3 = xofs[dx + 3] * 4;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2], alphap[4], alphap[4], alphap[4], alphap[4], alphap[6], alphap[6], alphap[6], alphap[6]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3], alphap[5], alphap[5], alphap[5], alphap[5], alphap[7], alphap[7], alphap[7], alphap[7]);

                __m128 _S00_0 = _mm_load_ps(S0 + sx0);
                __m128 _S00_1 = _mm_load_ps(S0 + sx1);
                __m128 _S00_2 = _mm_load_ps(S0 + sx2);
                __m128 _S00_3 = _mm_load_ps(S0 + sx3);
                __m128 _S01_0 = _mm_load_ps(S0 + sx0 + 4);
                __m128 _S01_1 = _mm_load_ps(S0 + sx1 + 4);
                __m128 _S01_2 = _mm_load_ps(S0 + sx2 + 4);
                __m128 _S01_3 = _mm_load_ps(S0 + sx3 + 4);

                __m512 _S00 = combine4x4_ps(_S00_0, _S00_1, _S00_2, _S00_3);
                __m512 _S01 = combine4x4_ps(_S01_0, _S01_1, _S01_2, _S01_3);

                __m512 _rows0 = _mm512_mul_ps(_S00, _a0);
                _rows0 = _mm512_fmadd_ps(_S01, _a1, _rows0);
                _mm512_storeu_ps(rows0p + dx * 4, _rows0);

                __m128 _S10_0 = _mm_load_ps(S1 + sx0);
                __m128 _S10_1 = _mm_load_ps(S1 + sx1);
                __m128 _S10_2 = _mm_load_ps(S1 + sx2);
                __m128 _S10_3 = _mm_load_ps(S1 + sx3);
                __m128 _S11_0 = _mm_load_ps(S1 + sx0 + 4);
                __m128 _S11_1 = _mm_load_ps(S1 + sx1 + 4);
                __m128 _S11_2 = _mm_load_ps(S1 + sx2 + 4);
                __m128 _S11_3 = _mm_load_ps(S1 + sx3 + 4);

                __m512 _S10 = combine4x4_ps(_S10_0, _S10_1, _S10_2, _S10_3);
                __m512 _S11 = combine4x4_ps(_S11_0, _S11_1, _S11_2, _S11_3);

                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_storeu_ps(rows1p + dx * 4, _rows1);

                alphap += 8;
            }
#endif // __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 4;
                int sx1 = xofs[dx + 1] * 4;

                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3]);

                __m256 _S00 = combine4x2_ps(_mm_load_ps(S0 + sx0), _mm_load_ps(S0 + sx1));
                __m256 _S01 = combine4x2_ps(_mm_load_ps(S0 + sx0 + 4), _mm_load_ps(S0 + sx1 + 4));
                __m256 _S10 = combine4x2_ps(_mm_load_ps(S1 + sx0), _mm_load_ps(S1 + sx1));
                __m256 _S11 = combine4x2_ps(_mm_load_ps(S1 + sx0 + 4), _mm_load_ps(S1 + sx1 + 4));

                __m256 _rows0 = _mm256_mul_ps(_S00, _a0);
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows0 = _mm256_comp_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_storeu_ps(rows0p + dx * 4, _rows0);
                _mm256_storeu_ps(rows1p + dx * 4, _rows1);

                alphap += 4;
            }
#endif // __AVX__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);

                __m128 _S00 = _mm_load_ps(S0p);
                __m128 _S01 = _mm_load_ps(S0p + 4);
                __m128 _S10 = _mm_load_ps(S1p);
                __m128 _S11 = _mm_load_ps(S1p + 4);
                __m128 _rows0 = _mm_mul_ps(_S00, _a0);
                __m128 _rows1 = _mm_mul_ps(_S10, _a0);
                _rows0 = _mm_comp_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm_store_ps(rows0p + dx * 4, _rows0);
                _mm_store_ps(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bilinear(rows0, rows1, dst.row(dy), w * 4, beta[0], beta[1]);

        beta += 2;
    }
}

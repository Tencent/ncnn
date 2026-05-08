// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void resize_bilinear_image_pack8(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)8 * 4u, 8);
    Mat rowsbuf1(w, (size_t)8 * 4u, 8);
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
#if __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 8;
                int sx1 = xofs[dx + 1] * 8;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3]);

                __m512 _S10 = combine8x2_ps(_mm256_load_ps(S1 + sx0), _mm256_load_ps(S1 + sx1));
                __m512 _S11 = combine8x2_ps(_mm256_load_ps(S1 + sx0 + 8), _mm256_load_ps(S1 + sx1 + 8));

                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_storeu_ps(rows1p + dx * 8, _rows1);

                alphap += 4;
            }
#endif // __AVX512F__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const float* S1p = S1 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);

                __m256 _S10 = _mm256_load_ps(S1p);
                __m256 _S11 = _mm256_load_ps(S1p + 8);
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_store_ps(rows1p + dx * 8, _rows1);

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
#if __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 8;
                int sx1 = xofs[dx + 1] * 8;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3]);

                __m512 _S00 = combine8x2_ps(_mm256_load_ps(S0 + sx0), _mm256_load_ps(S0 + sx1));
                __m512 _S01 = combine8x2_ps(_mm256_load_ps(S0 + sx0 + 8), _mm256_load_ps(S0 + sx1 + 8));
                __m512 _S10 = combine8x2_ps(_mm256_load_ps(S1 + sx0), _mm256_load_ps(S1 + sx1));
                __m512 _S11 = combine8x2_ps(_mm256_load_ps(S1 + sx0 + 8), _mm256_load_ps(S1 + sx1 + 8));

                __m512 _rows0 = _mm512_mul_ps(_S00, _a0);
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows0 = _mm512_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_storeu_ps(rows0p + dx * 8, _rows0);
                _mm512_storeu_ps(rows1p + dx * 8, _rows1);

                alphap += 4;
            }
#endif // __AVX512F__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);

                __m256 _S00 = _mm256_load_ps(S0p);
                __m256 _S01 = _mm256_load_ps(S0p + 8);
                __m256 _S10 = _mm256_load_ps(S1p);
                __m256 _S11 = _mm256_load_ps(S1p + 8);
                __m256 _rows0 = _mm256_mul_ps(_S00, _a0);
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows0 = _mm256_comp_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_store_ps(rows0p + dx * 8, _rows0);
                _mm256_store_ps(rows1p + dx * 8, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bilinear(rows0, rows1, dst.row(dy), w * 8, beta[0], beta[1]);

        beta += 2;
    }
}

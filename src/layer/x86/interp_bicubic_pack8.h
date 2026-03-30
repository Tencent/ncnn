// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void resize_bicubic_image_pack8(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)8 * 4u, 8);
    Mat rowsbuf1(w, (size_t)8 * 4u, 8);
    Mat rowsbuf2(w, (size_t)8 * 4u, 8);
    Mat rowsbuf3(w, (size_t)8 * 4u, 8);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

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
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;
            int dx = 0;
#if __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 8;
                int sx1 = xofs[dx + 1] * 8;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5]);
                __m512 _a2 = _mm512_setr_ps(alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6]);
                __m512 _a3 = _mm512_setr_ps(alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7]);

                __m512 _S30 = combine8x2_ps(_mm256_load_ps(S3 + sx0 - 8), _mm256_load_ps(S3 + sx1 - 8));
                __m512 _S31 = combine8x2_ps(_mm256_load_ps(S3 + sx0), _mm256_load_ps(S3 + sx1));
                __m512 _S32 = combine8x2_ps(_mm256_load_ps(S3 + sx0 + 8), _mm256_load_ps(S3 + sx1 + 8));
                __m512 _S33 = combine8x2_ps(_mm256_load_ps(S3 + sx0 + 16), _mm256_load_ps(S3 + sx1 + 16));

                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_storeu_ps(rows3p + dx * 8, _rows3);

                alphap += 8;
            }
#endif // __AVX512F__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const float* S3p = S3 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);
                __m256 _a2 = _mm256_set1_ps(alphap[2]);
                __m256 _a3 = _mm256_set1_ps(alphap[3]);

                __m256 _S30 = _mm256_load_ps(S3p - 8);
                __m256 _S31 = _mm256_load_ps(S3p + 0);
                __m256 _S32 = _mm256_load_ps(S3p + 8);
                __m256 _S33 = _mm256_load_ps(S3p + 16);
                __m256 _rows3 = _mm256_mul_ps(_S30, _a0);
                _rows3 = _mm256_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm256_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm256_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm256_store_ps(rows3p + dx * 8, _rows3);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            int dx = 0;
#if __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 8;
                int sx1 = xofs[dx + 1] * 8;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5]);
                __m512 _a2 = _mm512_setr_ps(alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6]);
                __m512 _a3 = _mm512_setr_ps(alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7]);

                __m512 _S20 = combine8x2_ps(_mm256_load_ps(S2 + sx0 - 8), _mm256_load_ps(S2 + sx1 - 8));
                __m512 _S21 = combine8x2_ps(_mm256_load_ps(S2 + sx0), _mm256_load_ps(S2 + sx1));
                __m512 _S22 = combine8x2_ps(_mm256_load_ps(S2 + sx0 + 8), _mm256_load_ps(S2 + sx1 + 8));
                __m512 _S23 = combine8x2_ps(_mm256_load_ps(S2 + sx0 + 16), _mm256_load_ps(S2 + sx1 + 16));

                __m512 _rows2 = _mm512_mul_ps(_S20, _a0);
                _rows2 = _mm512_fmadd_ps(_S21, _a1, _rows2);
                _rows2 = _mm512_fmadd_ps(_S22, _a2, _rows2);
                _rows2 = _mm512_fmadd_ps(_S23, _a3, _rows2);
                _mm512_storeu_ps(rows2p + dx * 8, _rows2);

                __m512 _S30 = combine8x2_ps(_mm256_load_ps(S3 + sx0 - 8), _mm256_load_ps(S3 + sx1 - 8));
                __m512 _S31 = combine8x2_ps(_mm256_load_ps(S3 + sx0), _mm256_load_ps(S3 + sx1));
                __m512 _S32 = combine8x2_ps(_mm256_load_ps(S3 + sx0 + 8), _mm256_load_ps(S3 + sx1 + 8));
                __m512 _S33 = combine8x2_ps(_mm256_load_ps(S3 + sx0 + 16), _mm256_load_ps(S3 + sx1 + 16));

                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_storeu_ps(rows3p + dx * 8, _rows3);

                alphap += 8;
            }
#endif // __AVX512F__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);
                __m256 _a2 = _mm256_set1_ps(alphap[2]);
                __m256 _a3 = _mm256_set1_ps(alphap[3]);

                __m256 _S20 = _mm256_load_ps(S2p - 8);
                __m256 _S21 = _mm256_load_ps(S2p + 0);
                __m256 _S22 = _mm256_load_ps(S2p + 8);
                __m256 _S23 = _mm256_load_ps(S2p + 16);
                __m256 _S30 = _mm256_load_ps(S3p - 8);
                __m256 _S31 = _mm256_load_ps(S3p + 0);
                __m256 _S32 = _mm256_load_ps(S3p + 8);
                __m256 _S33 = _mm256_load_ps(S3p + 16);
                __m256 _rows2 = _mm256_mul_ps(_S20, _a0);
                __m256 _rows3 = _mm256_mul_ps(_S30, _a0);
                _rows2 = _mm256_comp_fmadd_ps(_S21, _a1, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows2 = _mm256_comp_fmadd_ps(_S22, _a2, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows2 = _mm256_comp_fmadd_ps(_S23, _a3, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm256_store_ps(rows2p + dx * 8, _rows2);
                _mm256_store_ps(rows3p + dx * 8, _rows3);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const float* S1 = src.row(sy);
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            int dx = 0;
#if __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 8;
                int sx1 = xofs[dx + 1] * 8;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5]);
                __m512 _a2 = _mm512_setr_ps(alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6]);
                __m512 _a3 = _mm512_setr_ps(alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7]);

                __m512 _S10 = combine8x2_ps(_mm256_load_ps(S1 + sx0 - 8), _mm256_load_ps(S1 + sx1 - 8));
                __m512 _S11 = combine8x2_ps(_mm256_load_ps(S1 + sx0), _mm256_load_ps(S1 + sx1));
                __m512 _S12 = combine8x2_ps(_mm256_load_ps(S1 + sx0 + 8), _mm256_load_ps(S1 + sx1 + 8));
                __m512 _S13 = combine8x2_ps(_mm256_load_ps(S1 + sx0 + 16), _mm256_load_ps(S1 + sx1 + 16));
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _rows1 = _mm512_fmadd_ps(_S12, _a2, _rows1);
                _rows1 = _mm512_fmadd_ps(_S13, _a3, _rows1);
                _mm512_storeu_ps(rows1p + dx * 8, _rows1);

                __m512 _S20 = combine8x2_ps(_mm256_load_ps(S2 + sx0 - 8), _mm256_load_ps(S2 + sx1 - 8));
                __m512 _S21 = combine8x2_ps(_mm256_load_ps(S2 + sx0), _mm256_load_ps(S2 + sx1));
                __m512 _S22 = combine8x2_ps(_mm256_load_ps(S2 + sx0 + 8), _mm256_load_ps(S2 + sx1 + 8));
                __m512 _S23 = combine8x2_ps(_mm256_load_ps(S2 + sx0 + 16), _mm256_load_ps(S2 + sx1 + 16));
                __m512 _rows2 = _mm512_mul_ps(_S20, _a0);
                _rows2 = _mm512_fmadd_ps(_S21, _a1, _rows2);
                _rows2 = _mm512_fmadd_ps(_S22, _a2, _rows2);
                _rows2 = _mm512_fmadd_ps(_S23, _a3, _rows2);
                _mm512_storeu_ps(rows2p + dx * 8, _rows2);

                __m512 _S30 = combine8x2_ps(_mm256_load_ps(S3 + sx0 - 8), _mm256_load_ps(S3 + sx1 - 8));
                __m512 _S31 = combine8x2_ps(_mm256_load_ps(S3 + sx0), _mm256_load_ps(S3 + sx1));
                __m512 _S32 = combine8x2_ps(_mm256_load_ps(S3 + sx0 + 8), _mm256_load_ps(S3 + sx1 + 8));
                __m512 _S33 = combine8x2_ps(_mm256_load_ps(S3 + sx0 + 16), _mm256_load_ps(S3 + sx1 + 16));
                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_storeu_ps(rows3p + dx * 8, _rows3);

                alphap += 8;
            }
#endif // __AVX512F__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);
                __m256 _a2 = _mm256_set1_ps(alphap[2]);
                __m256 _a3 = _mm256_set1_ps(alphap[3]);

                __m256 _S10 = _mm256_load_ps(S1p - 8);
                __m256 _S11 = _mm256_load_ps(S1p + 0);
                __m256 _S12 = _mm256_load_ps(S1p + 8);
                __m256 _S13 = _mm256_load_ps(S1p + 16);
                __m256 _S20 = _mm256_load_ps(S2p - 8);
                __m256 _S21 = _mm256_load_ps(S2p + 0);
                __m256 _S22 = _mm256_load_ps(S2p + 8);
                __m256 _S23 = _mm256_load_ps(S2p + 16);
                __m256 _S30 = _mm256_load_ps(S3p - 8);
                __m256 _S31 = _mm256_load_ps(S3p + 0);
                __m256 _S32 = _mm256_load_ps(S3p + 8);
                __m256 _S33 = _mm256_load_ps(S3p + 16);
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                __m256 _rows2 = _mm256_mul_ps(_S20, _a0);
                __m256 _rows3 = _mm256_mul_ps(_S30, _a0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _rows2 = _mm256_comp_fmadd_ps(_S21, _a1, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows1 = _mm256_comp_fmadd_ps(_S12, _a2, _rows1);
                _rows2 = _mm256_comp_fmadd_ps(_S22, _a2, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows1 = _mm256_comp_fmadd_ps(_S13, _a3, _rows1);
                _rows2 = _mm256_comp_fmadd_ps(_S23, _a3, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm256_store_ps(rows1p + dx * 8, _rows1);
                _mm256_store_ps(rows2p + dx * 8, _rows2);
                _mm256_store_ps(rows3p + dx * 8, _rows3);

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const float* S0 = src.row(sy - 1);
            const float* S1 = src.row(sy);
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            int dx = 0;
#if __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 8;
                int sx1 = xofs[dx + 1] * 8;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[0], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4], alphap[4]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[1], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5], alphap[5]);
                __m512 _a2 = _mm512_setr_ps(alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[2], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6], alphap[6]);
                __m512 _a3 = _mm512_setr_ps(alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[3], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7], alphap[7]);

                __m512 _S00 = combine8x2_ps(_mm256_load_ps(S0 + sx0 - 8), _mm256_load_ps(S0 + sx1 - 8));
                __m512 _S01 = combine8x2_ps(_mm256_load_ps(S0 + sx0), _mm256_load_ps(S0 + sx1));
                __m512 _S02 = combine8x2_ps(_mm256_load_ps(S0 + sx0 + 8), _mm256_load_ps(S0 + sx1 + 8));
                __m512 _S03 = combine8x2_ps(_mm256_load_ps(S0 + sx0 + 16), _mm256_load_ps(S0 + sx1 + 16));
                __m512 _rows0 = _mm512_mul_ps(_S00, _a0);
                _rows0 = _mm512_fmadd_ps(_S01, _a1, _rows0);
                _rows0 = _mm512_fmadd_ps(_S02, _a2, _rows0);
                _rows0 = _mm512_fmadd_ps(_S03, _a3, _rows0);
                _mm512_storeu_ps(rows0p + dx * 8, _rows0);

                __m512 _S10 = combine8x2_ps(_mm256_load_ps(S1 + sx0 - 8), _mm256_load_ps(S1 + sx1 - 8));
                __m512 _S11 = combine8x2_ps(_mm256_load_ps(S1 + sx0), _mm256_load_ps(S1 + sx1));
                __m512 _S12 = combine8x2_ps(_mm256_load_ps(S1 + sx0 + 8), _mm256_load_ps(S1 + sx1 + 8));
                __m512 _S13 = combine8x2_ps(_mm256_load_ps(S1 + sx0 + 16), _mm256_load_ps(S1 + sx1 + 16));
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _rows1 = _mm512_fmadd_ps(_S12, _a2, _rows1);
                _rows1 = _mm512_fmadd_ps(_S13, _a3, _rows1);
                _mm512_storeu_ps(rows1p + dx * 8, _rows1);

                __m512 _S20 = combine8x2_ps(_mm256_load_ps(S2 + sx0 - 8), _mm256_load_ps(S2 + sx1 - 8));
                __m512 _S21 = combine8x2_ps(_mm256_load_ps(S2 + sx0), _mm256_load_ps(S2 + sx1));
                __m512 _S22 = combine8x2_ps(_mm256_load_ps(S2 + sx0 + 8), _mm256_load_ps(S2 + sx1 + 8));
                __m512 _S23 = combine8x2_ps(_mm256_load_ps(S2 + sx0 + 16), _mm256_load_ps(S2 + sx1 + 16));
                __m512 _rows2 = _mm512_mul_ps(_S20, _a0);
                _rows2 = _mm512_fmadd_ps(_S21, _a1, _rows2);
                _rows2 = _mm512_fmadd_ps(_S22, _a2, _rows2);
                _rows2 = _mm512_fmadd_ps(_S23, _a3, _rows2);
                _mm512_storeu_ps(rows2p + dx * 8, _rows2);

                __m512 _S30 = combine8x2_ps(_mm256_load_ps(S3 + sx0 - 8), _mm256_load_ps(S3 + sx1 - 8));
                __m512 _S31 = combine8x2_ps(_mm256_load_ps(S3 + sx0), _mm256_load_ps(S3 + sx1));
                __m512 _S32 = combine8x2_ps(_mm256_load_ps(S3 + sx0 + 8), _mm256_load_ps(S3 + sx1 + 8));
                __m512 _S33 = combine8x2_ps(_mm256_load_ps(S3 + sx0 + 16), _mm256_load_ps(S3 + sx1 + 16));
                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_storeu_ps(rows3p + dx * 8, _rows3);

                alphap += 8;
            }
#endif // __AVX512F__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);
                __m256 _a2 = _mm256_set1_ps(alphap[2]);
                __m256 _a3 = _mm256_set1_ps(alphap[3]);

                __m256 _S00 = _mm256_load_ps(S0p - 8);
                __m256 _S01 = _mm256_load_ps(S0p + 0);
                __m256 _S02 = _mm256_load_ps(S0p + 8);
                __m256 _S03 = _mm256_load_ps(S0p + 16);
                __m256 _S10 = _mm256_load_ps(S1p - 8);
                __m256 _S11 = _mm256_load_ps(S1p + 0);
                __m256 _S12 = _mm256_load_ps(S1p + 8);
                __m256 _S13 = _mm256_load_ps(S1p + 16);
                __m256 _S20 = _mm256_load_ps(S2p - 8);
                __m256 _S21 = _mm256_load_ps(S2p + 0);
                __m256 _S22 = _mm256_load_ps(S2p + 8);
                __m256 _S23 = _mm256_load_ps(S2p + 16);
                __m256 _S30 = _mm256_load_ps(S3p - 8);
                __m256 _S31 = _mm256_load_ps(S3p + 0);
                __m256 _S32 = _mm256_load_ps(S3p + 8);
                __m256 _S33 = _mm256_load_ps(S3p + 16);
                __m256 _rows0 = _mm256_mul_ps(_S00, _a0);
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                __m256 _rows2 = _mm256_mul_ps(_S20, _a0);
                __m256 _rows3 = _mm256_mul_ps(_S30, _a0);
                _rows0 = _mm256_comp_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _rows2 = _mm256_comp_fmadd_ps(_S21, _a1, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows0 = _mm256_comp_fmadd_ps(_S02, _a2, _rows0);
                _rows1 = _mm256_comp_fmadd_ps(_S12, _a2, _rows1);
                _rows2 = _mm256_comp_fmadd_ps(_S22, _a2, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows0 = _mm256_comp_fmadd_ps(_S03, _a3, _rows0);
                _rows1 = _mm256_comp_fmadd_ps(_S13, _a3, _rows1);
                _rows2 = _mm256_comp_fmadd_ps(_S23, _a3, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm256_store_ps(rows0p + dx * 8, _rows0);
                _mm256_store_ps(rows1p + dx * 8, _rows1);
                _mm256_store_ps(rows2p + dx * 8, _rows2);
                _mm256_store_ps(rows3p + dx * 8, _rows3);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bicubic(rows0, rows1, rows2, rows3, dst.row(dy), w * 8, beta[0], beta[1], beta[2], beta[3]);

        beta += 4;
    }
}

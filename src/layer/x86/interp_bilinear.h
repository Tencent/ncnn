// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void resize_bilinear_image_avx2(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs);
#endif

static void linear_coeffs(int w, int outw, int* xofs, float* alpha, int align_corner)
{
    double scale = (double)w / outw;
    if (align_corner)
    {
        scale = (double)(w - 1) / (outw - 1);
    }

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = (float)(dx * scale);
        }

        int sx = (int)floorf(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

static void vresize_bilinear(const float* rows0, const float* rows1, float* Dp, int n, float b0, float b1)
{
    int nn = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _b0_512 = _mm512_set1_ps(b0);
    __m512 _b1_512 = _mm512_set1_ps(b1);
    for (; nn + 15 < n; nn += 16)
    {
        __m512 _rows0 = _mm512_loadu_ps(rows0 + nn);
        __m512 _rows1 = _mm512_loadu_ps(rows1 + nn);
        __m512 _Dp = _mm512_mul_ps(_rows0, _b0_512);
        _Dp = _mm512_fmadd_ps(_rows1, _b1_512, _Dp);
        _mm512_storeu_ps(Dp + nn, _Dp);
    }
#endif // __AVX512F__
    __m256 _b0_256 = _mm256_set1_ps(b0);
    __m256 _b1_256 = _mm256_set1_ps(b1);
    for (; nn + 7 < n; nn += 8)
    {
        __m256 _rows0 = _mm256_loadu_ps(rows0 + nn);
        __m256 _rows1 = _mm256_loadu_ps(rows1 + nn);
        __m256 _Dp = _mm256_mul_ps(_rows0, _b0_256);
        _Dp = _mm256_comp_fmadd_ps(_rows1, _b1_256, _Dp);
        _mm256_storeu_ps(Dp + nn, _Dp);
    }
#endif // __AVX__
    __m128 _b0_128 = _mm_set1_ps(b0);
    __m128 _b1_128 = _mm_set1_ps(b1);
    for (; nn + 3 < n; nn += 4)
    {
        __m128 _rows0 = _mm_loadu_ps(rows0 + nn);
        __m128 _rows1 = _mm_loadu_ps(rows1 + nn);
        __m128 _Dp = _mm_mul_ps(_rows0, _b0_128);
        _Dp = _mm_comp_fmadd_ps(_rows1, _b1_128, _Dp);
        _mm_storeu_ps(Dp + nn, _Dp);
    }
#endif // __SSE2__
    for (; nn < n; nn++)
    {
        Dp[nn] = rows0[nn] * b0 + rows1[nn] * b1;
    }
}

static void resize_bilinear_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        resize_bilinear_image_avx2(src, dst, alpha, xofs, beta, yofs);
        return;
    }
#endif

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; dx + 15 < w; dx += 16)
            {
                __m512i _sx = _mm512_loadu_si512(xofs + dx);
                __m512i _sx1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(1));

                __m512 _S10 = _mm512_i32gather_ps(_sx, S1, sizeof(float));
                __m512 _S11 = _mm512_i32gather_ps(_sx1, S1, sizeof(float));

                __m512i _alpha_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
                __m512 _a0 = _mm512_i32gather_ps(_alpha_idx, alphap, sizeof(float));
                __m512i _alpha_idx1 = _mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(1));
                __m512 _a1 = _mm512_i32gather_ps(_alpha_idx1, alphap, sizeof(float));

                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_storeu_ps(rows1p + dx, _rows1);

                alphap += 32;
            }
#endif // __AVX512F__
            for (; dx + 7 < w; dx += 8)
            {
#if __AVX2__
                __m256i _sx = _mm256_loadu_si256((const __m256i*)(xofs + dx));
                __m256i _sx1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(1));

                __m256 _S10 = _mm256_i32gather_ps(S1, _sx, sizeof(float));
                __m256 _S11 = _mm256_i32gather_ps(S1, _sx1, sizeof(float));

                __m256i _alpha_idx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
                __m256 _a0 = _mm256_i32gather_ps(alphap, _alpha_idx, sizeof(float));
                __m256i _alpha_idx1 = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
                __m256 _a1 = _mm256_i32gather_ps(alphap, _alpha_idx1, sizeof(float));
#else
                __m256 _S10 = _mm256_setr_ps(S1[xofs[dx]], S1[xofs[dx + 1]], S1[xofs[dx + 2]], S1[xofs[dx + 3]], S1[xofs[dx + 4]], S1[xofs[dx + 5]], S1[xofs[dx + 6]], S1[xofs[dx + 7]]);
                __m256 _S11 = _mm256_setr_ps(S1[xofs[dx] + 1], S1[xofs[dx + 1] + 1], S1[xofs[dx + 2] + 1], S1[xofs[dx + 3] + 1], S1[xofs[dx + 4] + 1], S1[xofs[dx + 5] + 1], S1[xofs[dx + 6] + 1], S1[xofs[dx + 7] + 1]);

                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[2], alphap[4], alphap[6], alphap[8], alphap[10], alphap[12], alphap[14]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[3], alphap[5], alphap[7], alphap[9], alphap[11], alphap[13], alphap[15]);
#endif

                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_storeu_ps(rows1p + dx, _rows1);

                alphap += 16;
            }
#endif // __AVX__
            for (; dx + 3 < w; dx += 4)
            {
                __m128 _S10 = _mm_setr_ps(S1[xofs[dx]], S1[xofs[dx + 1]], S1[xofs[dx + 2]], S1[xofs[dx + 3]]);
                __m128 _S11 = _mm_setr_ps(S1[xofs[dx] + 1], S1[xofs[dx + 1] + 1], S1[xofs[dx + 2] + 1], S1[xofs[dx + 3] + 1]);

                __m128 _a01 = _mm_loadu_ps(alphap);
                __m128 _a23 = _mm_loadu_ps(alphap + 4);

                __m128 _a0 = _mm_shuffle_ps(_a01, _a23, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 _a1 = _mm_shuffle_ps(_a01, _a23, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 _rows1 = _mm_mul_ps(_S10, _a0);
                _rows1 = _mm_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm_storeu_ps(rows1p + dx, _rows1);

                alphap += 8;
            }
#endif // __SSE2__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

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
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; dx + 15 < w; dx += 16)
            {
                __m512i _sx = _mm512_loadu_si512(xofs + dx);
                __m512i _sx1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(1));

                __m512 _S00 = _mm512_i32gather_ps(_sx, S0, sizeof(float));
                __m512 _S01 = _mm512_i32gather_ps(_sx1, S0, sizeof(float));
                __m512 _S10 = _mm512_i32gather_ps(_sx, S1, sizeof(float));
                __m512 _S11 = _mm512_i32gather_ps(_sx1, S1, sizeof(float));

                __m512i _alpha_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
                __m512 _a0 = _mm512_i32gather_ps(_alpha_idx, alphap, sizeof(float));
                __m512i _alpha_idx1 = _mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(1));
                __m512 _a1 = _mm512_i32gather_ps(_alpha_idx1, alphap, sizeof(float));

                __m512 _rows0 = _mm512_mul_ps(_S00, _a0);
                _rows0 = _mm512_fmadd_ps(_S01, _a1, _rows0);
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_storeu_ps(rows0p + dx, _rows0);
                _mm512_storeu_ps(rows1p + dx, _rows1);

                alphap += 32;
            }
#endif // __AVX512F__
            for (; dx + 7 < w; dx += 8)
            {
#if __AVX2__
                __m256i _sx = _mm256_loadu_si256((const __m256i*)(xofs + dx));
                __m256i _sx1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(1));

                __m256 _S00 = _mm256_i32gather_ps(S0, _sx, sizeof(float));
                __m256 _S01 = _mm256_i32gather_ps(S0, _sx1, sizeof(float));
                __m256 _S10 = _mm256_i32gather_ps(S1, _sx, sizeof(float));
                __m256 _S11 = _mm256_i32gather_ps(S1, _sx1, sizeof(float));

                __m256i _alpha_idx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
                __m256 _a0 = _mm256_i32gather_ps(alphap, _alpha_idx, sizeof(float));
                __m256i _alpha_idx1 = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
                __m256 _a1 = _mm256_i32gather_ps(alphap, _alpha_idx1, sizeof(float));
#else
                __m256 _S00 = _mm256_setr_ps(S0[xofs[dx]], S0[xofs[dx + 1]], S0[xofs[dx + 2]], S0[xofs[dx + 3]], S0[xofs[dx + 4]], S0[xofs[dx + 5]], S0[xofs[dx + 6]], S0[xofs[dx + 7]]);
                __m256 _S01 = _mm256_setr_ps(S0[xofs[dx] + 1], S0[xofs[dx + 1] + 1], S0[xofs[dx + 2] + 1], S0[xofs[dx + 3] + 1], S0[xofs[dx + 4] + 1], S0[xofs[dx + 5] + 1], S0[xofs[dx + 6] + 1], S0[xofs[dx + 7] + 1]);
                __m256 _S10 = _mm256_setr_ps(S1[xofs[dx]], S1[xofs[dx + 1]], S1[xofs[dx + 2]], S1[xofs[dx + 3]], S1[xofs[dx + 4]], S1[xofs[dx + 5]], S1[xofs[dx + 6]], S1[xofs[dx + 7]]);
                __m256 _S11 = _mm256_setr_ps(S1[xofs[dx] + 1], S1[xofs[dx + 1] + 1], S1[xofs[dx + 2] + 1], S1[xofs[dx + 3] + 1], S1[xofs[dx + 4] + 1], S1[xofs[dx + 5] + 1], S1[xofs[dx + 6] + 1], S1[xofs[dx + 7] + 1]);

                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[2], alphap[4], alphap[6], alphap[8], alphap[10], alphap[12], alphap[14]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[3], alphap[5], alphap[7], alphap[9], alphap[11], alphap[13], alphap[15]);
#endif

                __m256 _rows0 = _mm256_mul_ps(_S00, _a0);
                _rows0 = _mm256_comp_fmadd_ps(_S01, _a1, _rows0);
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_storeu_ps(rows0p + dx, _rows0);
                _mm256_storeu_ps(rows1p + dx, _rows1);

                alphap += 16;
            }
#endif // __AVX__
            for (; dx + 3 < w; dx += 4)
            {
                __m128 _S00 = _mm_setr_ps(S0[xofs[dx]], S0[xofs[dx + 1]], S0[xofs[dx + 2]], S0[xofs[dx + 3]]);
                __m128 _S01 = _mm_setr_ps(S0[xofs[dx] + 1], S0[xofs[dx + 1] + 1], S0[xofs[dx + 2] + 1], S0[xofs[dx + 3] + 1]);
                __m128 _S10 = _mm_setr_ps(S1[xofs[dx]], S1[xofs[dx + 1]], S1[xofs[dx + 2]], S1[xofs[dx + 3]]);
                __m128 _S11 = _mm_setr_ps(S1[xofs[dx] + 1], S1[xofs[dx + 1] + 1], S1[xofs[dx + 2] + 1], S1[xofs[dx + 3] + 1]);

                __m128 _a01 = _mm_loadu_ps(alphap);
                __m128 _a23 = _mm_loadu_ps(alphap + 4);

                __m128 _a0 = _mm_shuffle_ps(_a01, _a23, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 _a1 = _mm_shuffle_ps(_a01, _a23, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 _rows0 = _mm_mul_ps(_S00, _a0);
                _rows0 = _mm_comp_fmadd_ps(_S01, _a1, _rows0);
                __m128 _rows1 = _mm_mul_ps(_S10, _a0);
                _rows1 = _mm_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm_storeu_ps(rows0p + dx, _rows0);
                _mm_storeu_ps(rows1p + dx, _rows1);

                alphap += 8;
            }
#endif // __SSE2__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bilinear(rows0, rows1, dst.row(dy), w, beta[0], beta[1]);

        beta += 2;
    }
}

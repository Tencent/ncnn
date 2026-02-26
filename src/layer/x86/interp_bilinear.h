// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

static void resize_bilinear_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
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
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

        int dx = 0;
#if __SSE2__
#if __AVX__
        __m256 _b0_256 = _mm256_set1_ps(b0);
        __m256 _b1_256 = _mm256_set1_ps(b1);
        for (; dx + 7 < w; dx += 8)
        {
            __m256 _rows0 = _mm256_loadu_ps(rows0p);
            __m256 _rows1 = _mm256_loadu_ps(rows1p);
            __m256 _Dp = _mm256_mul_ps(_rows0, _b0_256);
            _Dp = _mm256_comp_fmadd_ps(_rows1, _b1_256, _Dp);
            _mm256_storeu_ps(Dp, _Dp);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#endif // __AVX__
        __m128 _b0_128 = _mm_set1_ps(b0);
        __m128 _b1_128 = _mm_set1_ps(b1);
        for (; dx + 3 < w; dx += 4)
        {
            __m128 _rows0 = _mm_loadu_ps(rows0p);
            __m128 _rows1 = _mm_loadu_ps(rows1p);
            __m128 _Dp = _mm_mul_ps(_rows0, _b0_128);
            _Dp = _mm_comp_fmadd_ps(_rows1, _b1_128, _Dp);
            _mm_storeu_ps(Dp, _Dp);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }
#endif // __SSE2__
        for (; dx < w; dx++)
        {
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
}

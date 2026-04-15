// Copyright 2024 Tencent
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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const float* S1p = S1 + sx;

                __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);

                __m256 _S10 = (__m256)__lasx_xvld(S1p, 0);
                __m256 _S11 = (__m256)__lasx_xvld(S1p + 8, 0);
                __m256 _rows1 = __lasx_xvfmul_s(_S10, _a0);
                _rows1 = __lasx_xvfmadd_s(_a1, _S11, _rows1);
                __lasx_xvst(_rows1, rows1p + dx * 8, 0);

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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);

                __m256 _S00 = (__m256)__lasx_xvld(S0p, 0);
                __m256 _S01 = (__m256)__lasx_xvld(S0p + 8, 0);
                __m256 _S10 = (__m256)__lasx_xvld(S1p, 0);
                __m256 _S11 = (__m256)__lasx_xvld(S1p + 8, 0);
                __m256 _rows0 = __lasx_xvfmul_s(_S00, _a0);
                __m256 _rows1 = __lasx_xvfmul_s(_S10, _a0);
                _rows0 = __lasx_xvfmadd_s(_a1, _S01, _rows0);
                _rows1 = __lasx_xvfmadd_s(_a1, _S11, _rows1);
                __lasx_xvst(_rows0, rows0p + dx * 8, 0);
                __lasx_xvst(_rows1, rows1p + dx * 8, 0);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        __m256 _b0 = __lasx_xvreplfr2vr_s(beta[0]);
        __m256 _b1 = __lasx_xvreplfr2vr_s(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            __m256 _rows0 = (__m256)__lasx_xvld(rows0p, 0);
            __m256 _rows1 = (__m256)__lasx_xvld(rows1p, 0);
            __m256 _Dp = __lasx_xvfmul_s(_rows0, _b0);
            _Dp = __lasx_xvfmadd_s(_b1, _rows1, _Dp);
            __lasx_xvst(_Dp, Dp, 0);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }

        beta += 2;
    }
}

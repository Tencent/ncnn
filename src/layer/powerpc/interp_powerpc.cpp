// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "interp_powerpc.h"

#include <algorithm>
#include <math.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

#if __ALTIVEC__

// NOTE gcc altivec auto-vectorization may produce wrong results for the
// coefficient loops (the same issue worked around in interp.cpp), keep them scalar
__attribute__((optimize("no-tree-vectorize")))
#endif
static void
linear_coeffs(int w, int outw, double scale, int* xofs, float* alpha, int align_corner)
{
    // preserve the existing linear/bilinear same-size axis convention
    if (w == outw)
        scale = 1.0;

    if (align_corner)
    {
        scale = outw > 1 ? (double)(w - 1) / (outw - 1) : 0.0;
    }

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = static_cast<float>(dx * scale);
        }

        int sx = static_cast<int>(floor(fx));
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

static inline void interpolate_cubic(float fx, float* coeffs)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs[0] = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs[1] = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs[2] = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static void cubic_coeffs(int w, int outw, double scale, int* xofs, float* alpha, int align_corner)
{
    if (align_corner)
    {
        scale = outw > 1 ? (double)(w - 1) / (outw - 1) : 0.0;
    }

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = static_cast<float>(dx * scale);
        }

        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        interpolate_cubic(fx, alpha + dx * 4);

        if (sx <= -1)
        {
            sx = 1;
            alpha[dx * 4 + 0] = 1.f - alpha[dx * 4 + 3];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = 0.f;
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == 0)
        {
            sx = 1;
            alpha[dx * 4 + 0] = alpha[dx * 4 + 0] + alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 2];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == w - 2)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = alpha[dx * 4 + 2] + alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 0] = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = 1.f - alpha[dx * 4 + 0];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 1] = 0.f;
            alpha[dx * 4 + 0] = 0.f;
        }

        xofs[dx] = sx;
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
            for (int dx = 0; dx < w; dx++)
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
            for (int dx = 0; dx < w; dx++)
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

        int i = 0;
#if __ALTIVEC__
        v4f32 _b0 = __ncnn_altivec_fill_f32(b0);
        v4f32 _b1 = __ncnn_altivec_fill_f32(b1);
        for (; i + 3 < w; i += 4)
        {
            v4f32 _rows0 = __ncnn_altivec_load(rows0p);
            v4f32 _rows1 = __ncnn_altivec_load(rows1p);

            v4f32 _Dp = __ncnn_altivec_mul(_rows0, _b0);
            _Dp = __ncnn_altivec_fmadd(_Dp, _rows1, _b1);

            __ncnn_altivec_store(Dp, _Dp);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }
#endif // __ALTIVEC__
        for (; i < w; i++)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
}

static void resize_bicubic_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    Mat rowsbuf2(w);
    Mat rowsbuf3(w);
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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows0p[dx] = S0p[-1] * a0 + S0p[0] * a1 + S0p[1] * a2 + S0p[2] * a3;
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];
        float b2 = beta[2];
        float b3 = beta[3];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst.row(dy);

        int i = 0;
#if __ALTIVEC__
        v4f32 _b0 = __ncnn_altivec_fill_f32(b0);
        v4f32 _b1 = __ncnn_altivec_fill_f32(b1);
        v4f32 _b2 = __ncnn_altivec_fill_f32(b2);
        v4f32 _b3 = __ncnn_altivec_fill_f32(b3);
        for (; i + 3 < w; i += 4)
        {
            v4f32 _rows0 = __ncnn_altivec_load(rows0p);
            v4f32 _rows1 = __ncnn_altivec_load(rows1p);
            v4f32 _rows2 = __ncnn_altivec_load(rows2p);
            v4f32 _rows3 = __ncnn_altivec_load(rows3p);

            v4f32 _Dp = __ncnn_altivec_mul(_rows0, _b0);
            _Dp = __ncnn_altivec_fmadd(_Dp, _rows1, _b1);
            _Dp = __ncnn_altivec_fmadd(_Dp, _rows2, _b2);
            _Dp = __ncnn_altivec_fmadd(_Dp, _rows3, _b3);

            __ncnn_altivec_store(Dp, _Dp);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
            rows2p += 4;
            rows3p += 4;
        }
#endif // __ALTIVEC__
        for (; i < w; i++)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1 + rows2[x]*b2 + rows3[x]*b3;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1 + *rows2p++ * b2 + *rows3p++ * b3;
        }

        beta += 4;
    }
}

#if __ALTIVEC__
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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S1p = S1 + sx;

                v4f32 _a0 = __ncnn_altivec_fill_f32(alphap[0]);
                v4f32 _a1 = __ncnn_altivec_fill_f32(alphap[1]);

                v4f32 _S10 = __ncnn_altivec_load(S1p);
                v4f32 _S11 = __ncnn_altivec_load(S1p + 4);
                v4f32 _rows1 = __ncnn_altivec_mul(_S10, _a0);
                _rows1 = __ncnn_altivec_fmadd(_rows1, _S11, _a1);
                __ncnn_altivec_store(rows1p + dx * 4, _rows1);

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
                int sx = xofs[dx] * 4;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                v4f32 _a0 = __ncnn_altivec_fill_f32(alphap[0]);
                v4f32 _a1 = __ncnn_altivec_fill_f32(alphap[1]);

                v4f32 _S00 = __ncnn_altivec_load(S0p);
                v4f32 _S01 = __ncnn_altivec_load(S0p + 4);
                v4f32 _S10 = __ncnn_altivec_load(S1p);
                v4f32 _S11 = __ncnn_altivec_load(S1p + 4);
                v4f32 _rows0 = __ncnn_altivec_mul(_S00, _a0);
                v4f32 _rows1 = __ncnn_altivec_mul(_S10, _a0);
                _rows0 = __ncnn_altivec_fmadd(_rows0, _S01, _a1);
                _rows1 = __ncnn_altivec_fmadd(_rows1, _S11, _a1);
                __ncnn_altivec_store(rows0p + dx * 4, _rows0);
                __ncnn_altivec_store(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        v4f32 _b0 = __ncnn_altivec_fill_f32(beta[0]);
        v4f32 _b1 = __ncnn_altivec_fill_f32(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            v4f32 _rows0 = __ncnn_altivec_load(rows0p);
            v4f32 _rows1 = __ncnn_altivec_load(rows1p);
            v4f32 _Dp = __ncnn_altivec_mul(_rows0, _b0);
            _Dp = __ncnn_altivec_fmadd(_Dp, _rows1, _b1);
            __ncnn_altivec_store(Dp, _Dp);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

static void resize_bicubic_image_pack4(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)4 * 4u, 4);
    Mat rowsbuf1(w, (size_t)4 * 4u, 4);
    Mat rowsbuf2(w, (size_t)4 * 4u, 4);
    Mat rowsbuf3(w, (size_t)4 * 4u, 4);
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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S3p = S3 + sx;

                v4f32 _a0 = __ncnn_altivec_fill_f32(alphap[0]);
                v4f32 _a1 = __ncnn_altivec_fill_f32(alphap[1]);
                v4f32 _a2 = __ncnn_altivec_fill_f32(alphap[2]);
                v4f32 _a3 = __ncnn_altivec_fill_f32(alphap[3]);

                v4f32 _S30 = __ncnn_altivec_load(S3p - 4);
                v4f32 _S31 = __ncnn_altivec_load(S3p + 0);
                v4f32 _S32 = __ncnn_altivec_load(S3p + 4);
                v4f32 _S33 = __ncnn_altivec_load(S3p + 8);
                v4f32 _rows3 = __ncnn_altivec_mul(_S30, _a0);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S31, _a1);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S32, _a2);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S33, _a3);
                __ncnn_altivec_store(rows3p + dx * 4, _rows3);

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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                v4f32 _a0 = __ncnn_altivec_fill_f32(alphap[0]);
                v4f32 _a1 = __ncnn_altivec_fill_f32(alphap[1]);
                v4f32 _a2 = __ncnn_altivec_fill_f32(alphap[2]);
                v4f32 _a3 = __ncnn_altivec_fill_f32(alphap[3]);

                v4f32 _S20 = __ncnn_altivec_load(S2p - 4);
                v4f32 _S21 = __ncnn_altivec_load(S2p + 0);
                v4f32 _S22 = __ncnn_altivec_load(S2p + 4);
                v4f32 _S23 = __ncnn_altivec_load(S2p + 8);
                v4f32 _S30 = __ncnn_altivec_load(S3p - 4);
                v4f32 _S31 = __ncnn_altivec_load(S3p + 0);
                v4f32 _S32 = __ncnn_altivec_load(S3p + 4);
                v4f32 _S33 = __ncnn_altivec_load(S3p + 8);
                v4f32 _rows2 = __ncnn_altivec_mul(_S20, _a0);
                v4f32 _rows3 = __ncnn_altivec_mul(_S30, _a0);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S21, _a1);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S31, _a1);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S22, _a2);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S32, _a2);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S23, _a3);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S33, _a3);
                __ncnn_altivec_store(rows2p + dx * 4, _rows2);
                __ncnn_altivec_store(rows3p + dx * 4, _rows3);

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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                v4f32 _a0 = __ncnn_altivec_fill_f32(alphap[0]);
                v4f32 _a1 = __ncnn_altivec_fill_f32(alphap[1]);
                v4f32 _a2 = __ncnn_altivec_fill_f32(alphap[2]);
                v4f32 _a3 = __ncnn_altivec_fill_f32(alphap[3]);

                v4f32 _S10 = __ncnn_altivec_load(S1p - 4);
                v4f32 _S11 = __ncnn_altivec_load(S1p + 0);
                v4f32 _S12 = __ncnn_altivec_load(S1p + 4);
                v4f32 _S13 = __ncnn_altivec_load(S1p + 8);
                v4f32 _S20 = __ncnn_altivec_load(S2p - 4);
                v4f32 _S21 = __ncnn_altivec_load(S2p + 0);
                v4f32 _S22 = __ncnn_altivec_load(S2p + 4);
                v4f32 _S23 = __ncnn_altivec_load(S2p + 8);
                v4f32 _S30 = __ncnn_altivec_load(S3p - 4);
                v4f32 _S31 = __ncnn_altivec_load(S3p + 0);
                v4f32 _S32 = __ncnn_altivec_load(S3p + 4);
                v4f32 _S33 = __ncnn_altivec_load(S3p + 8);
                v4f32 _rows1 = __ncnn_altivec_mul(_S10, _a0);
                v4f32 _rows2 = __ncnn_altivec_mul(_S20, _a0);
                v4f32 _rows3 = __ncnn_altivec_mul(_S30, _a0);
                _rows1 = __ncnn_altivec_fmadd(_rows1, _S11, _a1);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S21, _a1);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S31, _a1);
                _rows1 = __ncnn_altivec_fmadd(_rows1, _S12, _a2);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S22, _a2);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S32, _a2);
                _rows1 = __ncnn_altivec_fmadd(_rows1, _S13, _a3);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S23, _a3);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S33, _a3);
                __ncnn_altivec_store(rows1p + dx * 4, _rows1);
                __ncnn_altivec_store(rows2p + dx * 4, _rows2);
                __ncnn_altivec_store(rows3p + dx * 4, _rows3);

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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                v4f32 _a0 = __ncnn_altivec_fill_f32(alphap[0]);
                v4f32 _a1 = __ncnn_altivec_fill_f32(alphap[1]);
                v4f32 _a2 = __ncnn_altivec_fill_f32(alphap[2]);
                v4f32 _a3 = __ncnn_altivec_fill_f32(alphap[3]);

                v4f32 _S00 = __ncnn_altivec_load(S0p - 4);
                v4f32 _S01 = __ncnn_altivec_load(S0p + 0);
                v4f32 _S02 = __ncnn_altivec_load(S0p + 4);
                v4f32 _S03 = __ncnn_altivec_load(S0p + 8);
                v4f32 _S10 = __ncnn_altivec_load(S1p - 4);
                v4f32 _S11 = __ncnn_altivec_load(S1p + 0);
                v4f32 _S12 = __ncnn_altivec_load(S1p + 4);
                v4f32 _S13 = __ncnn_altivec_load(S1p + 8);
                v4f32 _S20 = __ncnn_altivec_load(S2p - 4);
                v4f32 _S21 = __ncnn_altivec_load(S2p + 0);
                v4f32 _S22 = __ncnn_altivec_load(S2p + 4);
                v4f32 _S23 = __ncnn_altivec_load(S2p + 8);
                v4f32 _S30 = __ncnn_altivec_load(S3p - 4);
                v4f32 _S31 = __ncnn_altivec_load(S3p + 0);
                v4f32 _S32 = __ncnn_altivec_load(S3p + 4);
                v4f32 _S33 = __ncnn_altivec_load(S3p + 8);
                v4f32 _rows0 = __ncnn_altivec_mul(_S00, _a0);
                v4f32 _rows1 = __ncnn_altivec_mul(_S10, _a0);
                v4f32 _rows2 = __ncnn_altivec_mul(_S20, _a0);
                v4f32 _rows3 = __ncnn_altivec_mul(_S30, _a0);
                _rows0 = __ncnn_altivec_fmadd(_rows0, _S01, _a1);
                _rows1 = __ncnn_altivec_fmadd(_rows1, _S11, _a1);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S21, _a1);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S31, _a1);
                _rows0 = __ncnn_altivec_fmadd(_rows0, _S02, _a2);
                _rows1 = __ncnn_altivec_fmadd(_rows1, _S12, _a2);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S22, _a2);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S32, _a2);
                _rows0 = __ncnn_altivec_fmadd(_rows0, _S03, _a3);
                _rows1 = __ncnn_altivec_fmadd(_rows1, _S13, _a3);
                _rows2 = __ncnn_altivec_fmadd(_rows2, _S23, _a3);
                _rows3 = __ncnn_altivec_fmadd(_rows3, _S33, _a3);
                __ncnn_altivec_store(rows0p + dx * 4, _rows0);
                __ncnn_altivec_store(rows1p + dx * 4, _rows1);
                __ncnn_altivec_store(rows2p + dx * 4, _rows2);
                __ncnn_altivec_store(rows3p + dx * 4, _rows3);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        v4f32 _b0 = __ncnn_altivec_fill_f32(beta[0]);
        v4f32 _b1 = __ncnn_altivec_fill_f32(beta[1]);
        v4f32 _b2 = __ncnn_altivec_fill_f32(beta[2]);
        v4f32 _b3 = __ncnn_altivec_fill_f32(beta[3]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            v4f32 _rows0 = __ncnn_altivec_load(rows0p);
            v4f32 _rows1 = __ncnn_altivec_load(rows1p);
            v4f32 _rows2 = __ncnn_altivec_load(rows2p);
            v4f32 _rows3 = __ncnn_altivec_load(rows3p);
            v4f32 _Dp = __ncnn_altivec_mul(_rows0, _b0);
            _Dp = __ncnn_altivec_fmadd(_Dp, _rows1, _b1);
            _Dp = __ncnn_altivec_fmadd(_Dp, _rows2, _b2);
            _Dp = __ncnn_altivec_fmadd(_Dp, _rows3, _b3);
            __ncnn_altivec_store(Dp, _Dp);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
            rows2p += 4;
            rows3p += 4;
        }

        beta += 4;
    }
}
#endif // __ALTIVEC__

Interp_powerpc::Interp_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int Interp_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (!size_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_size_expr(bottom_blob_shapes, outw, outh);
    }

    if ((dims == 2 || dims == 3)
            && ((resize_type == 2 && (w == 1 || (dims == 3 && h == 1)))
                || (resize_type == 3 && (w < 4 || (dims == 3 && h < 4)))))
        return forward_small_input(bottom_blob, top_blob, outw, outh, opt);

#if __ALTIVEC__
    if (elempack == 4 && elemsize == 4u * 4)
    {
        if (dims == 1)
        {
            top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                v4f32 _v = __ncnn_altivec_load((const float*)bottom_blob + q * 4);
                float* outptr = top_blob_c;
                for (int i = 0; i < outw * outh; i++)
                {
                    __ncnn_altivec_store(outptr, _v);
                    outptr += 4;
                }
            }

            return 0;
        }

        if (dims == 2)
        {
            if (is_identity_resize(w, h, outw, outh, dims))
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (resize_type == 1) // nearest
            {
                const float ws = (output_width || dynamic_target_size || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        v4f32 _p = __ncnn_altivec_load(ptr + in_x * 4);
                        __ncnn_altivec_store(outptr, _p);

                        outptr += 4;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                const double ws = (output_width || dynamic_target_size || !size_expr.empty()) ? (double)w / outw : 1.0 / width_scale;
                linear_coeffs(w, outw, ws, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const float* Sp = ptr + sx;

                        v4f32 _a0 = __ncnn_altivec_fill_f32(alphap[0]);
                        v4f32 _a1 = __ncnn_altivec_fill_f32(alphap[1]);

                        v4f32 _S0 = __ncnn_altivec_load(Sp);
                        v4f32 _S1 = __ncnn_altivec_load(Sp + 4);
                        v4f32 _p = __ncnn_altivec_mul(_S0, _a0);
                        _p = __ncnn_altivec_fmadd(_p, _S1, _a1);
                        __ncnn_altivec_store(outptr, _p);

                        alphap += 2;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                const double ws = (output_width || dynamic_target_size || !size_expr.empty()) ? (double)w / outw : 1.0 / width_scale;
                cubic_coeffs(w, outw, ws, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const float* Sp = ptr + sx;

                        v4f32 _a0 = __ncnn_altivec_fill_f32(alphap[0]);
                        v4f32 _a1 = __ncnn_altivec_fill_f32(alphap[1]);
                        v4f32 _a2 = __ncnn_altivec_fill_f32(alphap[2]);
                        v4f32 _a3 = __ncnn_altivec_fill_f32(alphap[3]);

                        v4f32 _S0 = __ncnn_altivec_load(Sp - 4);
                        v4f32 _S1 = __ncnn_altivec_load(Sp + 0);
                        v4f32 _S2 = __ncnn_altivec_load(Sp + 4);
                        v4f32 _S3 = __ncnn_altivec_load(Sp + 8);
                        v4f32 _p = __ncnn_altivec_mul(_S0, _a0);
                        _p = __ncnn_altivec_fmadd(_p, _S1, _a1);
                        _p = __ncnn_altivec_fmadd(_p, _S2, _a2);
                        _p = __ncnn_altivec_fmadd(_p, _S3, _a3);
                        __ncnn_altivec_store(outptr, _p);

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }

        if (is_identity_resize(w, h, outw, outh, dims))
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (resize_type == 1) // nearest
        {
            const float hs = (output_height || dynamic_target_size || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
            const float ws = (output_width || dynamic_target_size || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const float* ptr = src.row(in_y);
                    float* outptr = dst.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        v4f32 _p = __ncnn_altivec_load(ptr + in_x * 4);
                        __ncnn_altivec_store(outptr, _p);

                        outptr += 4;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            const double ws = (output_width || dynamic_target_size || !size_expr.empty()) ? (double)w / outw : 1.0 / width_scale;
            linear_coeffs(w, outw, ws, xofs, alpha, align_corner);
            const double hs = (output_height || dynamic_target_size || !size_expr.empty()) ? (double)h / outh : 1.0 / height_scale;
            linear_coeffs(h, outh, hs, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

            const double ws = (output_width || dynamic_target_size || !size_expr.empty()) ? (double)w / outw : 1.0 / width_scale;
            cubic_coeffs(w, outw, ws, xofs, alpha, align_corner);
            const double hs = (output_height || dynamic_target_size || !size_expr.empty()) ? (double)h / outh : 1.0 / height_scale;
            cubic_coeffs(h, outh, hs, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __ALTIVEC__

    if (elempack != 1 || elemsize != 4u)
    {
        // non-fp32 or unsupported packing, delegate to the generic implementation
        Mat bottom_blob_unpacked;
        if (elempack != 1)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        }
        else
        {
            bottom_blob_unpacked = bottom_blob;
        }

        Mat top_blob_unpacked;
        int ret;
        if (elempack != 1)
        {
            std::vector<Mat> bottom_blobs_unpacked(2);
            bottom_blobs_unpacked[0] = bottom_blob_unpacked;
            bottom_blobs_unpacked[1] = reference_blob;

            std::vector<Mat> top_blobs_unpacked(1);
            ret = Interp::forward(bottom_blobs_unpacked, top_blobs_unpacked, opt);
            if (ret != 0)
                return ret;
            top_blob_unpacked = top_blobs_unpacked[0];
        }
        else
        {
            ret = Interp::forward(bottom_blobs, top_blobs, opt);
            return ret;
        }

        if (top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
        {
            Mat top_blob_packed;
            convert_packing(top_blob_unpacked, top_blob_packed, 4, opt);
            if (top_blob_packed.empty())
                return -100;
            top_blob = top_blob_packed;
        }
        else
        {
            top_blob = top_blob_unpacked;
        }

        return 0;
    }

    if (dims == 1)
    {
        // special case for 2d resize on flattened blob
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float v = bottom_blob[q];
            top_blob_c.fill(v);
        }

        return 0;
    }

    if (dims == 2)
    {
        if (is_identity_resize(w, h, outw, outh, dims))
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (resize_type == 1) // nearest
        {
            const float ws = (output_width || dynamic_target_size || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            const double ws = (output_width || dynamic_target_size || !size_expr.empty()) ? (double)w / outw : 1.0 / width_scale;
            linear_coeffs(w, outw, ws, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    *outptr++ = Sp[0] * a0 + Sp[1] * a1;
                    alphap += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            const double ws = (output_width || dynamic_target_size || !size_expr.empty()) ? (double)w / outw : 1.0 / width_scale;
            cubic_coeffs(w, outw, ws, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];
                    *outptr++ = Sp[-1] * a0 + Sp[0] * a1 + Sp[1] * a2 + Sp[2] * a3;
                    alphap += 4;
                }
            }

            delete[] buf;
        }

        return 0;
    }

    if (is_identity_resize(w, h, outw, outh, dims))
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (resize_type == 1) // nearest
    {
        const float hs = (output_height || dynamic_target_size || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
        const float ws = (output_width || dynamic_target_size || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_y * w + in_x];
                }
            }
        }
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        const double ws = (output_width || dynamic_target_size || !size_expr.empty()) ? (double)w / outw : 1.0 / width_scale;
        linear_coeffs(w, outw, ws, xofs, alpha, align_corner);
        const double hs = (output_height || dynamic_target_size || !size_expr.empty()) ? (double)h / outh : 1.0 / height_scale;
        linear_coeffs(h, outh, hs, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; ++q)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

        const double ws = (output_width || dynamic_target_size || !size_expr.empty()) ? (double)w / outw : 1.0 / width_scale;
        cubic_coeffs(w, outw, ws, xofs, alpha, align_corner);
        const double hs = (output_height || dynamic_target_size || !size_expr.empty()) ? (double)h / outh : 1.0 / height_scale;
        cubic_coeffs(h, outh, hs, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

} // namespace ncnn

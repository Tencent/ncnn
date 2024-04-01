// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static inline void interpolate_cubic_fp16sa(float fx, __fp16* coeffs)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs[0] = (__fp16)(A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A);
    coeffs[1] = (__fp16)((A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1);
    coeffs[2] = (__fp16)((A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1);
    coeffs[3] = (__fp16)((__fp16)1.f - coeffs[0] - coeffs[1] - coeffs[2]);
}

static void cubic_coeffs_fp16sa(int w, int outw, int* xofs, __fp16* alpha, int align_corner)
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
            fx = static_cast<float>(dx * scale);
        }

        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        interpolate_cubic_fp16sa(fx, alpha + dx * 4);

        if (sx <= -1)
        {
            sx = 1;
            alpha[dx * 4 + 0] = (__fp16)((__fp16)1.f - alpha[dx * 4 + 3]);
            alpha[dx * 4 + 1] = (__fp16)alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = (__fp16)0.f;
            alpha[dx * 4 + 3] = (__fp16)0.f;
        }
        if (sx == 0)
        {
            sx = 1;
            alpha[dx * 4 + 0] = (__fp16)(alpha[dx * 4 + 0] + alpha[dx * 4 + 1]);
            alpha[dx * 4 + 1] = (__fp16)alpha[dx * 4 + 2];
            alpha[dx * 4 + 2] = (__fp16)alpha[dx * 4 + 3];
            alpha[dx * 4 + 3] = (__fp16)0.f;
        }
        if (sx == w - 2)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = (__fp16)(alpha[dx * 4 + 2] + alpha[dx * 4 + 3]);
            alpha[dx * 4 + 2] = (__fp16)alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = (__fp16)alpha[dx * 4 + 0];
            alpha[dx * 4 + 0] = (__fp16)0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = (__fp16)((__fp16)1.f - alpha[dx * 4 + 0]);
            alpha[dx * 4 + 2] = (__fp16)(alpha[dx * 4 + 0]);
            alpha[dx * 4 + 1] = (__fp16)0.f;
            alpha[dx * 4 + 0] = (__fp16)0.f;
        }

        xofs[dx] = sx;
    }
}

static void resize_bicubic_image_fp16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
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
            const __fp16* S3 = src.row<const __fp16>(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows3p[dx] = (float)S3p[-1] * a0 + (float)S3p[0] * a1 + (float)S3p[1] * a2 + (float)S3p[2] * a3;

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
            const __fp16* S2 = src.row<const __fp16>(sy + 1);
            const __fp16* S3 = src.row<const __fp16>(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows2p[dx] = (float)S2p[-1] * a0 + (float)S2p[0] * a1 + (float)S2p[1] * a2 + (float)S2p[2] * a3;
                rows3p[dx] = (float)S3p[-1] * a0 + (float)S3p[0] * a1 + (float)S3p[1] * a2 + (float)S3p[2] * a3;

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
            const __fp16* S1 = src.row<const __fp16>(sy);
            const __fp16* S2 = src.row<const __fp16>(sy + 1);
            const __fp16* S3 = src.row<const __fp16>(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows1p[dx] = (float)S1p[-1] * a0 + (float)S1p[0] * a1 + (float)S1p[1] * a2 + (float)S1p[2] * a3;
                rows2p[dx] = (float)S2p[-1] * a0 + (float)S2p[0] * a1 + (float)S2p[1] * a2 + (float)S2p[2] * a3;
                rows3p[dx] = (float)S3p[-1] * a0 + (float)S3p[0] * a1 + (float)S3p[1] * a2 + (float)S3p[2] * a3;

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const __fp16* S0 = src.row<const __fp16>(sy - 1);
            const __fp16* S1 = src.row<const __fp16>(sy);
            const __fp16* S2 = src.row<const __fp16>(sy + 1);
            const __fp16* S3 = src.row<const __fp16>(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows0p[dx] = (float)S0p[-1] * a0 + (float)S0p[0] * a1 + (float)S0p[1] * a2 + (float)S0p[2] * a3;
                rows1p[dx] = (float)S1p[-1] * a0 + (float)S1p[0] * a1 + (float)S1p[1] * a2 + (float)S1p[2] * a3;
                rows2p[dx] = (float)S2p[-1] * a0 + (float)S2p[0] * a1 + (float)S2p[1] * a2 + (float)S2p[2] * a3;
                rows3p[dx] = (float)S3p[-1] * a0 + (float)S3p[0] * a1 + (float)S3p[1] * a2 + (float)S3p[2] * a3;

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
        __fp16* Dp = dst.row<__fp16>(dy);
        for (int dx = 0; dx < w; dx++)
        {
            // D[x] = rows0[x]*b0 + rows1[x]*b1 + rows2[x]*b2 + rows3[x]*b3;
            *Dp++ = (__fp16)(*rows0p++ * b0 + *rows1p++ * b1 + *rows2p++ * b2 + *rows3p++ * b3);
        }

        beta += 4;
    }
}

static void resize_bicubic_image_fp16sa(const Mat& src, Mat& dst, __fp16* alpha, int* xofs, __fp16* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)2u);
    Mat rowsbuf1(w, (size_t)2u);
    Mat rowsbuf2(w, (size_t)2u);
    Mat rowsbuf3(w, (size_t)2u);
    __fp16* rows0 = rowsbuf0;
    __fp16* rows1 = rowsbuf1;
    __fp16* rows2 = rowsbuf2;
    __fp16* rows3 = rowsbuf3;

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
            __fp16* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const __fp16* S3 = src.row<const __fp16>(sy + 2);

            const __fp16* alphap = alpha;
            __fp16* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S3p = S3 + sx;

                __fp16 a0 = alphap[0];
                __fp16 a1 = alphap[1];
                __fp16 a2 = alphap[2];
                __fp16 a3 = alphap[3];
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            __fp16* rows0_old = rows0;
            __fp16* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const __fp16* S2 = src.row<const __fp16>(sy + 1);
            const __fp16* S3 = src.row<const __fp16>(sy + 2);

            const __fp16* alphap = alpha;
            __fp16* rows2p = rows2;
            __fp16* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                __fp16 a0 = alphap[0];
                __fp16 a1 = alphap[1];
                __fp16 a2 = alphap[2];
                __fp16 a3 = alphap[3];
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            __fp16* rows0_old = rows0;
            __fp16* rows1_old = rows1;
            __fp16* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const __fp16* S1 = src.row<const __fp16>(sy);
            const __fp16* S2 = src.row<const __fp16>(sy + 1);
            const __fp16* S3 = src.row<const __fp16>(sy + 2);

            const __fp16* alphap = alpha;
            __fp16* rows1p = rows1;
            __fp16* rows2p = rows2;
            __fp16* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                __fp16 a0 = alphap[0];
                __fp16 a1 = alphap[1];
                __fp16 a2 = alphap[2];
                __fp16 a3 = alphap[3];
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const __fp16* S0 = src.row<const __fp16>(sy - 1);
            const __fp16* S1 = src.row<const __fp16>(sy);
            const __fp16* S2 = src.row<const __fp16>(sy + 1);
            const __fp16* S3 = src.row<const __fp16>(sy + 2);

            const __fp16* alphap = alpha;
            __fp16* rows0p = rows0;
            __fp16* rows1p = rows1;
            __fp16* rows2p = rows2;
            __fp16* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                __fp16 a0 = alphap[0];
                __fp16 a1 = alphap[1];
                __fp16 a2 = alphap[2];
                __fp16 a3 = alphap[3];
                rows0p[dx] = S0p[-1] * a0 + S0p[0] * a1 + S0p[1] * a2 + S0p[2] * a3;
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        __fp16 b0 = beta[0];
        __fp16 b1 = beta[1];
        __fp16 b2 = beta[2];
        __fp16 b3 = beta[3];

        __fp16* rows0p = rows0;
        __fp16* rows1p = rows1;
        __fp16* rows2p = rows2;
        __fp16* rows3p = rows3;
        __fp16* Dp = dst.row<__fp16>(dy);
        for (int dx = 0; dx < w; dx++)
        {
            // D[x] = rows0[x]*b0 + rows1[x]*b1 + rows2[x]*b2 + rows3[x]*b3;
            *Dp++ = (*rows0p++ * b0 + *rows1p++ * b1 + *rows2p++ * b2 + *rows3p++ * b3);
        }

        beta += 4;
    }
}

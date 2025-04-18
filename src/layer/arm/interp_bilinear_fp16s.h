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

static void linear_coeffs_fp16sa(int w, int outw, int* xofs, __fp16* alpha, int align_corner)
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

        int sx = floor(fx);
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

        alpha[dx * 2] = (__fp16)(1.f - fx);
        alpha[dx * 2 + 1] = (__fp16)fx;
    }
}

static void resize_bilinear_image_fp16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
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
            const __fp16* S1 = src.row<const __fp16>(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = (float)S1p[0] * a0 + (float)S1p[1] * a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const __fp16* S0 = src.row<const __fp16>(sy);
            const __fp16* S1 = src.row<const __fp16>(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = (float)S0p[0] * a0 + (float)S0p[1] * a1;
                rows1p[dx] = (float)S1p[0] * a0 + (float)S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        __fp16* Dp = dst.row<__fp16>(dy);

        int nn = w >> 3;
        int remain = w - (nn << 3);

        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        for (; nn > 0; nn--)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);

            float32x4_t _Dp = vmulq_f32(_rows0, _b0);
            _Dp = vfmaq_f32(_Dp, _rows1, _b1);

            vst1_f16(Dp, vcvt_f16_f32(_Dp));

            float32x4_t _rows0n = vld1q_f32(rows0p + 4);
            float32x4_t _rows1n = vld1q_f32(rows1p + 4);

            float32x4_t _Dn = vmulq_f32(_rows0n, _b0);
            _Dn = vfmaq_f32(_Dn, _rows1n, _b1);

            vst1_f16(Dp + 4, vcvt_f16_f32(_Dn));

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
        for (; remain; --remain)
        {
            // D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = (__fp16)(*rows0p++ * b0 + *rows1p++ * b1);
        }

        beta += 2;
    }
}

static void resize_bilinear_image_fp16sa(const Mat& src, Mat& dst, __fp16* alpha, int* xofs, __fp16* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)2u);
    Mat rowsbuf1(w, (size_t)2u);
    __fp16* rows0 = rowsbuf0;
    __fp16* rows1 = rowsbuf1;

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
            __fp16* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const __fp16* S1 = src.row<const __fp16>(sy + 1);

            const __fp16* alphap = alpha;
            __fp16* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S1p = S1 + sx;

                __fp16 a0 = alphap[0];
                __fp16 a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const __fp16* S0 = src.row<const __fp16>(sy);
            const __fp16* S1 = src.row<const __fp16>(sy + 1);

            const __fp16* alphap = alpha;
            __fp16* rows0p = rows0;
            __fp16* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;

                __fp16 a0 = alphap[0];
                __fp16 a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        __fp16 b0 = beta[0];
        __fp16 b1 = beta[1];

        __fp16* rows0p = rows0;
        __fp16* rows1p = rows1;
        __fp16* Dp = dst.row<__fp16>(dy);

        int nn = w >> 3;
        int remain = w - (nn << 3);

        float16x8_t _b0 = vdupq_n_f16(b0);
        float16x8_t _b1 = vdupq_n_f16(b1);
        for (; nn > 0; nn--)
        {
            float16x8_t _rows0 = vld1q_f16(rows0p);
            float16x8_t _rows1 = vld1q_f16(rows1p);

            float16x8_t _Dp = vmulq_f16(_rows0, _b0);
            _Dp = vfmaq_f16(_Dp, _rows1, _b1);

            vst1q_f16(Dp, _Dp);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
        for (; remain; --remain)
        {
            // D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = (__fp16)(*rows0p++ * b0 + *rows1p++ * b1);
        }

        beta += 2;
    }
}

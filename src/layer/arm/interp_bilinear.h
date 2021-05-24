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
#if __ARM_NEON
            for (; dx + 1 < w; dx += 2)
            {
                int sx = xofs[dx];
                int sxn = xofs[dx + 1];
                const float* S1p = S1 + sx;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }
#endif // __ARM_NEON
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
#if __ARM_NEON
            for (; dx + 1 < w; dx += 2)
            {
                int sx = xofs[dx];
                int sxn = xofs[dx + 1];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S0np = S0 + sxn;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S0 = vld1_f32(S0p);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S0n = vld1_f32(S0np);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S0S0n = vcombine_f32(_S0, _S0n);
                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms0 = vmulq_f32(_S0S0n, _a);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows0p + dx, _rows0);
                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }
#endif // __ARM_NEON
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

#if __ARM_NEON
        int nn = w >> 3;
#else
        int nn = 0;
#endif
        int remain = w - (nn << 3);

#if __ARM_NEON
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        for (; nn > 0; nn--)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);

            float32x4_t _D = vmulq_f32(_rows0, _b0);
            _D = vmlaq_f32(_D, _rows1, _b1);

            vst1q_f32(Dp, _D);

            float32x4_t _rows0n = vld1q_f32(rows0p + 4);
            float32x4_t _rows1n = vld1q_f32(rows1p + 4);

            float32x4_t _Dn = vmulq_f32(_rows0n, _b0);
            _Dn = vmlaq_f32(_Dn, _rows1n, _b1);

            vst1q_f32(Dp + 4, _Dn);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#endif // __ARM_NEON
        for (; remain; --remain)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
}

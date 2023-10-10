// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void resize_bilinear_image_pack16(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)16 * 4u, 16);
    Mat rowsbuf1(w, (size_t)16 * 4u, 16);
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
                int sx = xofs[dx] * 16;
                const float* S1p = S1 + sx;

                __m512 _a0 = _mm512_set1_ps(alphap[0]);
                __m512 _a1 = _mm512_set1_ps(alphap[1]);

                __m512 _S10 = _mm512_load_ps(S1p);
                __m512 _S11 = _mm512_load_ps(S1p + 16);
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_store_ps(rows1p + dx * 16, _rows1);

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
                int sx = xofs[dx] * 16;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                __m512 _a0 = _mm512_set1_ps(alphap[0]);
                __m512 _a1 = _mm512_set1_ps(alphap[1]);

                __m512 _S00 = _mm512_load_ps(S0p);
                __m512 _S01 = _mm512_load_ps(S0p + 16);
                __m512 _S10 = _mm512_load_ps(S1p);
                __m512 _S11 = _mm512_load_ps(S1p + 16);
                __m512 _rows0 = _mm512_mul_ps(_S00, _a0);
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows0 = _mm512_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_store_ps(rows0p + dx * 16, _rows0);
                _mm512_store_ps(rows1p + dx * 16, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        __m512 _b0 = _mm512_set1_ps(beta[0]);
        __m512 _b1 = _mm512_set1_ps(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            __m512 _rows0 = _mm512_load_ps(rows0p);
            __m512 _rows1 = _mm512_load_ps(rows1p);
            __m512 _Dp = _mm512_mul_ps(_rows0, _b0);
            _Dp = _mm512_fmadd_ps(_rows1, _b1, _Dp);
            _mm512_store_ps(Dp, _Dp);

            Dp += 16;
            rows0p += 16;
            rows1p += 16;
        }

        beta += 2;
    }
}

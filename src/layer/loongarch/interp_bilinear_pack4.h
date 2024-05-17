// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S1p = S1 + sx;

                __m128 _a0 = __lsx_vreplfr2vr_s(alphap[0]);
                __m128 _a1 = __lsx_vreplfr2vr_s(alphap[1]);

                __m128 _S10 = (__m128)__lsx_vld(S1p, 0);
                __m128 _S11 = (__m128)__lsx_vld(S1p + 4, 0);
                __m128 _rows1 = __lsx_vfmul_s(_S10, _a0);
                _rows1 = __lsx_vfmadd_s(_a1, _S11, _rows1);
                __lsx_vst(_rows1, rows1p + dx * 4, 0);

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
                int sx = xofs[dx] * 4;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                __m128 _a0 = __lsx_vreplfr2vr_s(alphap[0]);
                __m128 _a1 = __lsx_vreplfr2vr_s(alphap[1]);

                __m128 _S00 = (__m128)__lsx_vld(S0p, 0);
                __m128 _S01 = (__m128)__lsx_vld(S0p + 4, 0);
                __m128 _S10 = (__m128)__lsx_vld(S1p, 0);
                __m128 _S11 = (__m128)__lsx_vld(S1p + 4, 0);
                __m128 _rows0 = __lsx_vfmul_s(_S00, _a0);
                __m128 _rows1 = __lsx_vfmul_s(_S10, _a0);
                _rows0 = __lsx_vfmadd_s(_a1, _S01, _rows0);
                _rows1 = __lsx_vfmadd_s(_a1, _S11, _rows1);
                __lsx_vst(_rows0, rows0p + dx * 4, 0);
                __lsx_vst(_rows1, rows1p + dx * 4, 0);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        __m128 _b0 = __lsx_vreplfr2vr_s(beta[0]);
        __m128 _b1 = __lsx_vreplfr2vr_s(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            __m128 _rows0 = (__m128)__lsx_vld(rows0p, 0);
            __m128 _rows1 = (__m128)__lsx_vld(rows1p, 0);
            __m128 _Dp = __lsx_vfmul_s(_rows0, _b0);
            _Dp = __lsx_vfmadd_s(_b1, _rows1, _Dp);
            __lsx_vst(_Dp, Dp, 0);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

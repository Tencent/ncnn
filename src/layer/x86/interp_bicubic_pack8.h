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
            for (int dx = 0; dx < w; dx++)
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
            for (int dx = 0; dx < w; dx++)
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
            for (int dx = 0; dx < w; dx++)
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
            for (int dx = 0; dx < w; dx++)
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
        __m256 _b0 = _mm256_set1_ps(beta[0]);
        __m256 _b1 = _mm256_set1_ps(beta[1]);
        __m256 _b2 = _mm256_set1_ps(beta[2]);
        __m256 _b3 = _mm256_set1_ps(beta[3]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            __m256 _rows0 = _mm256_load_ps(rows0p);
            __m256 _rows1 = _mm256_load_ps(rows1p);
            __m256 _rows2 = _mm256_load_ps(rows2p);
            __m256 _rows3 = _mm256_load_ps(rows3p);
            __m256 _D = _mm256_mul_ps(_rows0, _b0);
            _D = _mm256_comp_fmadd_ps(_rows1, _b1, _D);
            _D = _mm256_comp_fmadd_ps(_rows2, _b2, _D);
            _D = _mm256_comp_fmadd_ps(_rows3, _b3, _D);
            _mm256_store_ps(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
            rows2p += 8;
            rows3p += 8;
        }

        beta += 4;
    }
}

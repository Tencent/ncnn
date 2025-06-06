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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);

                v4f32 _S10 = (v4f32)__msa_ld_w(S1p, 0);
                v4f32 _S11 = (v4f32)__msa_ld_w(S1p + 4, 0);
                v4f32 _rows1 = __msa_fmul_w(_S10, _a0);
                _rows1 = __msa_fmadd_w(_rows1, _S11, _a1);
                __msa_st_w((v4i32)_rows1, rows1p + dx * 4, 0);

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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);

                v4f32 _S00 = (v4f32)__msa_ld_w(S0p, 0);
                v4f32 _S01 = (v4f32)__msa_ld_w(S0p + 4, 0);
                v4f32 _S10 = (v4f32)__msa_ld_w(S1p, 0);
                v4f32 _S11 = (v4f32)__msa_ld_w(S1p + 4, 0);
                v4f32 _rows0 = __msa_fmul_w(_S00, _a0);
                v4f32 _rows1 = __msa_fmul_w(_S10, _a0);
                _rows0 = __msa_fmadd_w(_rows0, _S01, _a1);
                _rows1 = __msa_fmadd_w(_rows1, _S11, _a1);
                __msa_st_w((v4i32)_rows0, rows0p + dx * 4, 0);
                __msa_st_w((v4i32)_rows1, rows1p + dx * 4, 0);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        v4f32 _b0 = __msa_fill_w_f32(beta[0]);
        v4f32 _b1 = __msa_fill_w_f32(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            v4f32 _rows0 = (v4f32)__msa_ld_w(rows0p, 0);
            v4f32 _rows1 = (v4f32)__msa_ld_w(rows1p, 0);
            v4f32 _Dp = __msa_fmul_w(_rows0, _b0);
            _Dp = __msa_fmadd_w(_Dp, _rows1, _b1);
            __msa_st_w((v4i32)_Dp, Dp, 0);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

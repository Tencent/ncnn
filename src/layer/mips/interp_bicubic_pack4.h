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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                v4f32 _S30 = (v4f32)__msa_ld_w(S3p - 4, 0);
                v4f32 _S31 = (v4f32)__msa_ld_w(S3p + 0, 0);
                v4f32 _S32 = (v4f32)__msa_ld_w(S3p + 4, 0);
                v4f32 _S33 = (v4f32)__msa_ld_w(S3p + 8, 0);
                v4f32 _rows3 = __msa_fmul_w(_S30, _a0);
                _rows3 = __msa_fmadd_w(_rows3, _S31, _a1);
                _rows3 = __msa_fmadd_w(_rows3, _S32, _a2);
                _rows3 = __msa_fmadd_w(_rows3, _S33, _a3);
                __msa_st_w((v4i32)_rows3, rows3p + dx * 4, 0);

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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                v4f32 _S20 = (v4f32)__msa_ld_w(S2p - 4, 0);
                v4f32 _S21 = (v4f32)__msa_ld_w(S2p + 0, 0);
                v4f32 _S22 = (v4f32)__msa_ld_w(S2p + 4, 0);
                v4f32 _S23 = (v4f32)__msa_ld_w(S2p + 8, 0);
                v4f32 _S30 = (v4f32)__msa_ld_w(S3p - 4, 0);
                v4f32 _S31 = (v4f32)__msa_ld_w(S3p + 0, 0);
                v4f32 _S32 = (v4f32)__msa_ld_w(S3p + 4, 0);
                v4f32 _S33 = (v4f32)__msa_ld_w(S3p + 8, 0);
                v4f32 _rows2 = __msa_fmul_w(_S20, _a0);
                v4f32 _rows3 = __msa_fmul_w(_S30, _a0);
                _rows2 = __msa_fmadd_w(_rows2, _S21, _a1);
                _rows3 = __msa_fmadd_w(_rows3, _S31, _a1);
                _rows2 = __msa_fmadd_w(_rows2, _S22, _a2);
                _rows3 = __msa_fmadd_w(_rows3, _S32, _a2);
                _rows2 = __msa_fmadd_w(_rows2, _S23, _a3);
                _rows3 = __msa_fmadd_w(_rows3, _S33, _a3);
                __msa_st_w((v4i32)_rows2, rows2p + dx * 4, 0);
                __msa_st_w((v4i32)_rows3, rows3p + dx * 4, 0);

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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                v4f32 _S10 = (v4f32)__msa_ld_w(S1p - 4, 0);
                v4f32 _S11 = (v4f32)__msa_ld_w(S1p + 0, 0);
                v4f32 _S12 = (v4f32)__msa_ld_w(S1p + 4, 0);
                v4f32 _S13 = (v4f32)__msa_ld_w(S1p + 8, 0);
                v4f32 _S20 = (v4f32)__msa_ld_w(S2p - 4, 0);
                v4f32 _S21 = (v4f32)__msa_ld_w(S2p + 0, 0);
                v4f32 _S22 = (v4f32)__msa_ld_w(S2p + 4, 0);
                v4f32 _S23 = (v4f32)__msa_ld_w(S2p + 8, 0);
                v4f32 _S30 = (v4f32)__msa_ld_w(S3p - 4, 0);
                v4f32 _S31 = (v4f32)__msa_ld_w(S3p + 0, 0);
                v4f32 _S32 = (v4f32)__msa_ld_w(S3p + 4, 0);
                v4f32 _S33 = (v4f32)__msa_ld_w(S3p + 8, 0);
                v4f32 _rows1 = __msa_fmul_w(_S10, _a0);
                v4f32 _rows2 = __msa_fmul_w(_S20, _a0);
                v4f32 _rows3 = __msa_fmul_w(_S30, _a0);
                _rows1 = __msa_fmadd_w(_rows1, _S11, _a1);
                _rows2 = __msa_fmadd_w(_rows2, _S21, _a1);
                _rows3 = __msa_fmadd_w(_rows3, _S31, _a1);
                _rows1 = __msa_fmadd_w(_rows1, _S12, _a2);
                _rows2 = __msa_fmadd_w(_rows2, _S22, _a2);
                _rows3 = __msa_fmadd_w(_rows3, _S32, _a2);
                _rows1 = __msa_fmadd_w(_rows1, _S13, _a3);
                _rows2 = __msa_fmadd_w(_rows2, _S23, _a3);
                _rows3 = __msa_fmadd_w(_rows3, _S33, _a3);
                __msa_st_w((v4i32)_rows1, rows1p + dx * 4, 0);
                __msa_st_w((v4i32)_rows2, rows2p + dx * 4, 0);
                __msa_st_w((v4i32)_rows3, rows3p + dx * 4, 0);

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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                v4f32 _S00 = (v4f32)__msa_ld_w(S0p - 4, 0);
                v4f32 _S01 = (v4f32)__msa_ld_w(S0p + 0, 0);
                v4f32 _S02 = (v4f32)__msa_ld_w(S0p + 4, 0);
                v4f32 _S03 = (v4f32)__msa_ld_w(S0p + 8, 0);
                v4f32 _S10 = (v4f32)__msa_ld_w(S1p - 4, 0);
                v4f32 _S11 = (v4f32)__msa_ld_w(S1p + 0, 0);
                v4f32 _S12 = (v4f32)__msa_ld_w(S1p + 4, 0);
                v4f32 _S13 = (v4f32)__msa_ld_w(S1p + 8, 0);
                v4f32 _S20 = (v4f32)__msa_ld_w(S2p - 4, 0);
                v4f32 _S21 = (v4f32)__msa_ld_w(S2p + 0, 0);
                v4f32 _S22 = (v4f32)__msa_ld_w(S2p + 4, 0);
                v4f32 _S23 = (v4f32)__msa_ld_w(S2p + 8, 0);
                v4f32 _S30 = (v4f32)__msa_ld_w(S3p - 4, 0);
                v4f32 _S31 = (v4f32)__msa_ld_w(S3p + 0, 0);
                v4f32 _S32 = (v4f32)__msa_ld_w(S3p + 4, 0);
                v4f32 _S33 = (v4f32)__msa_ld_w(S3p + 8, 0);
                v4f32 _rows0 = __msa_fmul_w(_S00, _a0);
                v4f32 _rows1 = __msa_fmul_w(_S10, _a0);
                v4f32 _rows2 = __msa_fmul_w(_S20, _a0);
                v4f32 _rows3 = __msa_fmul_w(_S30, _a0);
                _rows0 = __msa_fmadd_w(_rows0, _S01, _a1);
                _rows1 = __msa_fmadd_w(_rows1, _S11, _a1);
                _rows2 = __msa_fmadd_w(_rows2, _S21, _a1);
                _rows3 = __msa_fmadd_w(_rows3, _S31, _a1);
                _rows0 = __msa_fmadd_w(_rows0, _S02, _a2);
                _rows1 = __msa_fmadd_w(_rows1, _S12, _a2);
                _rows2 = __msa_fmadd_w(_rows2, _S22, _a2);
                _rows3 = __msa_fmadd_w(_rows3, _S32, _a2);
                _rows0 = __msa_fmadd_w(_rows0, _S03, _a3);
                _rows1 = __msa_fmadd_w(_rows1, _S13, _a3);
                _rows2 = __msa_fmadd_w(_rows2, _S23, _a3);
                _rows3 = __msa_fmadd_w(_rows3, _S33, _a3);
                __msa_st_w((v4i32)_rows0, rows0p + dx * 4, 0);
                __msa_st_w((v4i32)_rows1, rows1p + dx * 4, 0);
                __msa_st_w((v4i32)_rows2, rows2p + dx * 4, 0);
                __msa_st_w((v4i32)_rows3, rows3p + dx * 4, 0);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        v4f32 _b0 = __msa_fill_w_f32(beta[0]);
        v4f32 _b1 = __msa_fill_w_f32(beta[1]);
        v4f32 _b2 = __msa_fill_w_f32(beta[2]);
        v4f32 _b3 = __msa_fill_w_f32(beta[3]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            v4f32 _rows0 = (v4f32)__msa_ld_w(rows0p, 0);
            v4f32 _rows1 = (v4f32)__msa_ld_w(rows1p, 0);
            v4f32 _rows2 = (v4f32)__msa_ld_w(rows2p, 0);
            v4f32 _rows3 = (v4f32)__msa_ld_w(rows3p, 0);
            v4f32 _Dp = __msa_fmul_w(_rows0, _b0);
            _Dp = __msa_fmadd_w(_Dp, _rows1, _b1);
            _Dp = __msa_fmadd_w(_Dp, _rows2, _b2);
            _Dp = __msa_fmadd_w(_Dp, _rows3, _b3);
            __msa_st_w((v4i32)_Dp, Dp, 0);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
            rows2p += 4;
            rows3p += 4;
        }

        beta += 4;
    }
}

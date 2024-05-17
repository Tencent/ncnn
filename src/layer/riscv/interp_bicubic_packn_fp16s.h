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

static void resize_bicubic_image_packn_fp16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)packn * 4u, packn);
    Mat rowsbuf1(w, (size_t)packn * 4u, packn);
    Mat rowsbuf2(w, (size_t)packn * 4u, packn);
    Mat rowsbuf3(w, (size_t)packn * 4u, packn);
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
                int sx = xofs[dx] * packn;
                const __fp16* S3p = S3 + sx;

                vfloat16m1_t _S30 = vle16_v_f16m1(S3p - packn, vl);
                vfloat16m1_t _S31 = vle16_v_f16m1(S3p, vl);
                vfloat16m1_t _S32 = vle16_v_f16m1(S3p + packn, vl);
                vfloat16m1_t _S33 = vle16_v_f16m1(S3p + packn * 2, vl);
                vfloat32m2_t _rows3 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                vse32_v_f32m2(rows3p + dx * packn, _rows3, vl);

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
                int sx = xofs[dx] * packn;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                vfloat16m1_t _S20 = vle16_v_f16m1(S2p - packn, vl);
                vfloat16m1_t _S21 = vle16_v_f16m1(S2p, vl);
                vfloat16m1_t _S22 = vle16_v_f16m1(S2p + packn, vl);
                vfloat16m1_t _S23 = vle16_v_f16m1(S2p + packn * 2, vl);
                vfloat16m1_t _S30 = vle16_v_f16m1(S3p - packn, vl);
                vfloat16m1_t _S31 = vle16_v_f16m1(S3p, vl);
                vfloat16m1_t _S32 = vle16_v_f16m1(S3p + packn, vl);
                vfloat16m1_t _S33 = vle16_v_f16m1(S3p + packn * 2, vl);
                vfloat32m2_t _rows2 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m2_t _rows3 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                vse32_v_f32m2(rows2p + dx * packn, _rows2, vl);
                vse32_v_f32m2(rows3p + dx * packn, _rows3, vl);

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
                int sx = xofs[dx] * packn;
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                vfloat16m1_t _S10 = vle16_v_f16m1(S1p - packn, vl);
                vfloat16m1_t _S11 = vle16_v_f16m1(S1p, vl);
                vfloat16m1_t _S12 = vle16_v_f16m1(S1p + packn, vl);
                vfloat16m1_t _S13 = vle16_v_f16m1(S1p + packn * 2, vl);
                vfloat16m1_t _S20 = vle16_v_f16m1(S2p - packn, vl);
                vfloat16m1_t _S21 = vle16_v_f16m1(S2p, vl);
                vfloat16m1_t _S22 = vle16_v_f16m1(S2p + packn, vl);
                vfloat16m1_t _S23 = vle16_v_f16m1(S2p + packn * 2, vl);
                vfloat16m1_t _S30 = vle16_v_f16m1(S3p - packn, vl);
                vfloat16m1_t _S31 = vle16_v_f16m1(S3p, vl);
                vfloat16m1_t _S32 = vle16_v_f16m1(S3p + packn, vl);
                vfloat16m1_t _S33 = vle16_v_f16m1(S3p + packn * 2, vl);
                vfloat32m2_t _rows1 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S10, alphap[0], vl), alphap[1], _S11, vl), alphap[2], _S12, vl), alphap[3], _S13, vl);
                vfloat32m2_t _rows2 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m2_t _rows3 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                vse32_v_f32m2(rows1p + dx * packn, _rows1, vl);
                vse32_v_f32m2(rows2p + dx * packn, _rows2, vl);
                vse32_v_f32m2(rows3p + dx * packn, _rows3, vl);

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
                int sx = xofs[dx] * packn;
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                vfloat16m1_t _S00 = vle16_v_f16m1(S0p - packn, vl);
                vfloat16m1_t _S01 = vle16_v_f16m1(S0p, vl);
                vfloat16m1_t _S02 = vle16_v_f16m1(S0p + packn, vl);
                vfloat16m1_t _S03 = vle16_v_f16m1(S0p + packn * 2, vl);
                vfloat16m1_t _S10 = vle16_v_f16m1(S1p - packn, vl);
                vfloat16m1_t _S11 = vle16_v_f16m1(S1p, vl);
                vfloat16m1_t _S12 = vle16_v_f16m1(S1p + packn, vl);
                vfloat16m1_t _S13 = vle16_v_f16m1(S1p + packn * 2, vl);
                vfloat16m1_t _S20 = vle16_v_f16m1(S2p - packn, vl);
                vfloat16m1_t _S21 = vle16_v_f16m1(S2p, vl);
                vfloat16m1_t _S22 = vle16_v_f16m1(S2p + packn, vl);
                vfloat16m1_t _S23 = vle16_v_f16m1(S2p + packn * 2, vl);
                vfloat16m1_t _S30 = vle16_v_f16m1(S3p - packn, vl);
                vfloat16m1_t _S31 = vle16_v_f16m1(S3p, vl);
                vfloat16m1_t _S32 = vle16_v_f16m1(S3p + packn, vl);
                vfloat16m1_t _S33 = vle16_v_f16m1(S3p + packn * 2, vl);
                vfloat32m2_t _rows0 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S00, alphap[0], vl), alphap[1], _S01, vl), alphap[2], _S02, vl), alphap[3], _S03, vl);
                vfloat32m2_t _rows1 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S10, alphap[0], vl), alphap[1], _S11, vl), alphap[2], _S12, vl), alphap[3], _S13, vl);
                vfloat32m2_t _rows2 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m2_t _rows3 = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                vse32_v_f32m2(rows0p + dx * packn, _rows0, vl);
                vse32_v_f32m2(rows1p + dx * packn, _rows1, vl);
                vse32_v_f32m2(rows2p + dx * packn, _rows2, vl);
                vse32_v_f32m2(rows3p + dx * packn, _rows3, vl);

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
            vfloat32m2_t _rows0 = vle32_v_f32m2(rows0p, vl);
            vfloat32m2_t _rows1 = vle32_v_f32m2(rows1p, vl);
            vfloat32m2_t _rows2 = vle32_v_f32m2(rows2p, vl);
            vfloat32m2_t _rows3 = vle32_v_f32m2(rows3p, vl);

            vfloat32m2_t _Dp = vfmacc_vf_f32m2(vfmacc_vf_f32m2(vfmacc_vf_f32m2(vfmul_vf_f32m2(_rows0, b0, vl), b1, _rows1, vl), b2, _rows2, vl), b3, _rows3, vl);

            vse16_v_f16m1(Dp, vfncvt_f_f_w_f16m1(_Dp, vl), vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
            rows2p += packn;
            rows3p += packn;
        }

        beta += 4;
    }
}

static void resize_bicubic_image_packn_fp16sa(const Mat& src, Mat& dst, __fp16* alpha, int* xofs, __fp16* beta, int* yofs)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)packn * 2u, packn);
    Mat rowsbuf1(w, (size_t)packn * 2u, packn);
    Mat rowsbuf2(w, (size_t)packn * 2u, packn);
    Mat rowsbuf3(w, (size_t)packn * 2u, packn);
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
                int sx = xofs[dx] * packn;
                const __fp16* S3p = S3 + sx;

                vfloat16m1_t _S30 = vle16_v_f16m1(S3p - packn, vl);
                vfloat16m1_t _S31 = vle16_v_f16m1(S3p, vl);
                vfloat16m1_t _S32 = vle16_v_f16m1(S3p + packn, vl);
                vfloat16m1_t _S33 = vle16_v_f16m1(S3p + packn * 2, vl);
                vfloat16m1_t _rows3 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                vse16_v_f16m1(rows3p + dx * packn, _rows3, vl);

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
                int sx = xofs[dx] * packn;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                vfloat16m1_t _S20 = vle16_v_f16m1(S2p - packn, vl);
                vfloat16m1_t _S21 = vle16_v_f16m1(S2p, vl);
                vfloat16m1_t _S22 = vle16_v_f16m1(S2p + packn, vl);
                vfloat16m1_t _S23 = vle16_v_f16m1(S2p + packn * 2, vl);
                vfloat16m1_t _S30 = vle16_v_f16m1(S3p - packn, vl);
                vfloat16m1_t _S31 = vle16_v_f16m1(S3p, vl);
                vfloat16m1_t _S32 = vle16_v_f16m1(S3p + packn, vl);
                vfloat16m1_t _S33 = vle16_v_f16m1(S3p + packn * 2, vl);
                vfloat16m1_t _rows2 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat16m1_t _rows3 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                vse16_v_f16m1(rows2p + dx * packn, _rows2, vl);
                vse16_v_f16m1(rows3p + dx * packn, _rows3, vl);

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
                int sx = xofs[dx] * packn;
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                vfloat16m1_t _S10 = vle16_v_f16m1(S1p - packn, vl);
                vfloat16m1_t _S11 = vle16_v_f16m1(S1p, vl);
                vfloat16m1_t _S12 = vle16_v_f16m1(S1p + packn, vl);
                vfloat16m1_t _S13 = vle16_v_f16m1(S1p + packn * 2, vl);
                vfloat16m1_t _S20 = vle16_v_f16m1(S2p - packn, vl);
                vfloat16m1_t _S21 = vle16_v_f16m1(S2p, vl);
                vfloat16m1_t _S22 = vle16_v_f16m1(S2p + packn, vl);
                vfloat16m1_t _S23 = vle16_v_f16m1(S2p + packn * 2, vl);
                vfloat16m1_t _S30 = vle16_v_f16m1(S3p - packn, vl);
                vfloat16m1_t _S31 = vle16_v_f16m1(S3p, vl);
                vfloat16m1_t _S32 = vle16_v_f16m1(S3p + packn, vl);
                vfloat16m1_t _S33 = vle16_v_f16m1(S3p + packn * 2, vl);
                vfloat16m1_t _rows1 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S10, alphap[0], vl), alphap[1], _S11, vl), alphap[2], _S12, vl), alphap[3], _S13, vl);
                vfloat16m1_t _rows2 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat16m1_t _rows3 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                vse16_v_f16m1(rows1p + dx * packn, _rows1, vl);
                vse16_v_f16m1(rows2p + dx * packn, _rows2, vl);
                vse16_v_f16m1(rows3p + dx * packn, _rows3, vl);

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
                int sx = xofs[dx] * packn;
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                vfloat16m1_t _S00 = vle16_v_f16m1(S0p - packn, vl);
                vfloat16m1_t _S01 = vle16_v_f16m1(S0p, vl);
                vfloat16m1_t _S02 = vle16_v_f16m1(S0p + packn, vl);
                vfloat16m1_t _S03 = vle16_v_f16m1(S0p + packn * 2, vl);
                vfloat16m1_t _S10 = vle16_v_f16m1(S1p - packn, vl);
                vfloat16m1_t _S11 = vle16_v_f16m1(S1p, vl);
                vfloat16m1_t _S12 = vle16_v_f16m1(S1p + packn, vl);
                vfloat16m1_t _S13 = vle16_v_f16m1(S1p + packn * 2, vl);
                vfloat16m1_t _S20 = vle16_v_f16m1(S2p - packn, vl);
                vfloat16m1_t _S21 = vle16_v_f16m1(S2p, vl);
                vfloat16m1_t _S22 = vle16_v_f16m1(S2p + packn, vl);
                vfloat16m1_t _S23 = vle16_v_f16m1(S2p + packn * 2, vl);
                vfloat16m1_t _S30 = vle16_v_f16m1(S3p - packn, vl);
                vfloat16m1_t _S31 = vle16_v_f16m1(S3p, vl);
                vfloat16m1_t _S32 = vle16_v_f16m1(S3p + packn, vl);
                vfloat16m1_t _S33 = vle16_v_f16m1(S3p + packn * 2, vl);
                vfloat16m1_t _rows0 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S00, alphap[0], vl), alphap[1], _S01, vl), alphap[2], _S02, vl), alphap[3], _S03, vl);
                vfloat16m1_t _rows1 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S10, alphap[0], vl), alphap[1], _S11, vl), alphap[2], _S12, vl), alphap[3], _S13, vl);
                vfloat16m1_t _rows2 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat16m1_t _rows3 = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                vse16_v_f16m1(rows0p + dx * packn, _rows0, vl);
                vse16_v_f16m1(rows1p + dx * packn, _rows1, vl);
                vse16_v_f16m1(rows2p + dx * packn, _rows2, vl);
                vse16_v_f16m1(rows3p + dx * packn, _rows3, vl);

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
            vfloat16m1_t _rows0 = vle16_v_f16m1(rows0p, vl);
            vfloat16m1_t _rows1 = vle16_v_f16m1(rows1p, vl);
            vfloat16m1_t _rows2 = vle16_v_f16m1(rows2p, vl);
            vfloat16m1_t _rows3 = vle16_v_f16m1(rows3p, vl);

            vfloat16m1_t _Dp = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_rows0, b0, vl), b1, _rows1, vl), b2, _rows2, vl), b3, _rows3, vl);

            vse16_v_f16m1(Dp, _Dp, vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
            rows2p += packn;
            rows3p += packn;
        }

        beta += 4;
    }
}

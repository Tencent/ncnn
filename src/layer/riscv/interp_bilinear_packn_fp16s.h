// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void resize_bilinear_image_packn_fp16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)packn * 4u, packn);
    Mat rowsbuf1(w, (size_t)packn * 4u, packn);
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
                int sx = xofs[dx] * packn;
                const __fp16* S1p = S1 + sx;

                vfloat16m1_t _S10 = vle16_v_f16m1(S1p, vl);
                vfloat16m1_t _S11 = vle16_v_f16m1(S1p + packn, vl);
                vfloat32m2_t _rows1 = vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S10, alphap[0], vl), alphap[1], _S11, vl);

                vse32_v_f32m2(rows1p + dx * packn, _rows1, vl);

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
                int sx = xofs[dx] * packn;
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;

                vfloat16m1_t _S00 = vle16_v_f16m1(S0p, vl);
                vfloat16m1_t _S01 = vle16_v_f16m1(S0p + packn, vl);
                vfloat16m1_t _S10 = vle16_v_f16m1(S1p, vl);
                vfloat16m1_t _S11 = vle16_v_f16m1(S1p + packn, vl);
                vfloat32m2_t _rows0 = vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S00, alphap[0], vl), alphap[1], _S01, vl);
                vfloat32m2_t _rows1 = vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S10, alphap[0], vl), alphap[1], _S11, vl);

                vse32_v_f32m2(rows0p + dx * packn, _rows0, vl);
                vse32_v_f32m2(rows1p + dx * packn, _rows1, vl);

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

        for (int dx = 0; dx < w; dx++)
        {
            vfloat32m2_t _rows0 = vle32_v_f32m2(rows0p, vl);
            vfloat32m2_t _rows1 = vle32_v_f32m2(rows1p, vl);

            vfloat32m2_t _Dp = vfmacc_vf_f32m2(vfmul_vf_f32m2(_rows0, b0, vl), b1, _rows1, vl);

            vse16_v_f16m1(Dp, vfncvt_f_f_w_f16m1(_Dp, vl), vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
        }

        beta += 2;
    }
}

static void resize_bilinear_image_packn_fp16sa(const Mat& src, Mat& dst, __fp16* alpha, int* xofs, __fp16* beta, int* yofs)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)packn * 2u, packn);
    Mat rowsbuf1(w, (size_t)packn * 2u, packn);
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
                int sx = xofs[dx] * packn;
                const __fp16* S1p = S1 + sx;

                vfloat16m1_t _S10 = vle16_v_f16m1(S1p, vl);
                vfloat16m1_t _S11 = vle16_v_f16m1(S1p + packn, vl);
                vfloat16m1_t _rows1 = vfmacc_vf_f16m1(vfmul_vf_f16m1(_S10, alphap[0], vl), alphap[1], _S11, vl);

                vse16_v_f16m1(rows1p + dx * packn, _rows1, vl);

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
                int sx = xofs[dx] * packn;
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;

                vfloat16m1_t _S00 = vle16_v_f16m1(S0p, vl);
                vfloat16m1_t _S01 = vle16_v_f16m1(S0p + packn, vl);
                vfloat16m1_t _S10 = vle16_v_f16m1(S1p, vl);
                vfloat16m1_t _S11 = vle16_v_f16m1(S1p + packn, vl);
                vfloat16m1_t _rows0 = vfmacc_vf_f16m1(vfmul_vf_f16m1(_S00, alphap[0], vl), alphap[1], _S01, vl);
                vfloat16m1_t _rows1 = vfmacc_vf_f16m1(vfmul_vf_f16m1(_S10, alphap[0], vl), alphap[1], _S11, vl);

                vse16_v_f16m1(rows0p + dx * packn, _rows0, vl);
                vse16_v_f16m1(rows1p + dx * packn, _rows1, vl);

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

        for (int dx = 0; dx < w; dx++)
        {
            vfloat16m1_t _rows0 = vle16_v_f16m1(rows0p, vl);
            vfloat16m1_t _rows1 = vle16_v_f16m1(rows1p, vl);

            vfloat16m1_t _Dp = vfmacc_vf_f16m1(vfmul_vf_f16m1(_rows0, b0, vl), b1, _rows1, vl);

            vse16_v_f16m1(Dp, _Dp, vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
        }

        beta += 2;
    }
}

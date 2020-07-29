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

static void resize_bilinear_image_pack4_fp16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
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
            const __fp16* S1 = src.row<const __fp16>(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const __fp16* S1p = S1 + sx;

                float32x2_t _a01 = vld1_f32(alphap);

                float32x4_t _S10 = vcvt_f32_f16(vld1_f16(S1p));
                float32x4_t _S11 = vcvt_f32_f16(vld1_f16(S1p + 4));
                float32x4_t _rows1 = vmulq_lane_f32(_S10, _a01, 0);
                _rows1 = vmlaq_lane_f32(_rows1, _S11, _a01, 1);
                vst1q_f32(rows1p + dx * 4, _rows1);

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
                int sx = xofs[dx] * 4;
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;

                float32x2_t _a01 = vld1_f32(alphap);

                float32x4_t _S00 = vcvt_f32_f16(vld1_f16(S0p));
                float32x4_t _S01 = vcvt_f32_f16(vld1_f16(S0p + 4));
                float32x4_t _S10 = vcvt_f32_f16(vld1_f16(S1p));
                float32x4_t _S11 = vcvt_f32_f16(vld1_f16(S1p + 4));
                float32x4_t _rows0 = vmulq_lane_f32(_S00, _a01, 0);
                float32x4_t _rows1 = vmulq_lane_f32(_S10, _a01, 0);
                _rows0 = vmlaq_lane_f32(_rows0, _S01, _a01, 1);
                _rows1 = vmlaq_lane_f32(_rows1, _S11, _a01, 1);
                vst1q_f32(rows0p + dx * 4, _rows0);
                vst1q_f32(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float32x2_t _b01 = vld1_f32(beta);

        float* rows0p = rows0;
        float* rows1p = rows1;
        __fp16* Dp = dst.row<__fp16>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            float32x4_t _D = vmulq_lane_f32(_rows0, _b01, 0);
            _D = vmlaq_lane_f32(_D, _rows1, _b01, 1);
            vst1_f16(Dp, vcvt_f16_f32(_D));

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

static void resize_bilinear_image_pack4_fp16sa(const Mat& src, Mat& dst, __fp16* alpha, int* xofs, __fp16* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)4 * 2u, 4);
    Mat rowsbuf1(w, (size_t)4 * 2u, 4);
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
                int sx = xofs[dx] * 4;
                const __fp16* S1p = S1 + sx;

                float16x4_t _a01 = vld1_f16(alphap);

                float16x4_t _S10 = vld1_f16(S1p);
                float16x4_t _S11 = vld1_f16(S1p + 4);
                float16x4_t _rows1 = vmul_lane_f16(_S10, _a01, 0);
                _rows1 = vfma_lane_f16(_rows1, _S11, _a01, 1);
                vst1_f16(rows1p + dx * 4, _rows1);

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
                int sx = xofs[dx] * 4;
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;

                float16x4_t _a01 = vld1_f16(alphap);

                float16x4_t _S00 = vld1_f16(S0p);
                float16x4_t _S01 = vld1_f16(S0p + 4);
                float16x4_t _S10 = vld1_f16(S1p);
                float16x4_t _S11 = vld1_f16(S1p + 4);
                float16x4_t _rows0 = vmul_lane_f16(_S00, _a01, 0);
                float16x4_t _rows1 = vmul_lane_f16(_S10, _a01, 0);
                _rows0 = vfma_lane_f16(_rows0, _S01, _a01, 1);
                _rows1 = vfma_lane_f16(_rows1, _S11, _a01, 1);
                vst1_f16(rows0p + dx * 4, _rows0);
                vst1_f16(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float16x4_t _b01 = vld1_f16(beta);

        __fp16* rows0p = rows0;
        __fp16* rows1p = rows1;
        __fp16* Dp = dst.row<__fp16>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            float16x4_t _rows0 = vld1_f16(rows0p);
            float16x4_t _rows1 = vld1_f16(rows1p);
            float16x4_t _D = vmul_lane_f16(_rows0, _b01, 0);
            _D = vfma_lane_f16(_D, _rows1, _b01, 1);
            vst1_f16(Dp, _D);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

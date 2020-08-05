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

static void resize_bicubic_image_pack4_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
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
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S3p = S3 + sx;

                float32x4_t _a0123 = vld1q_f32(alphap);

                float32x4_t _S30 = vcvt_f32_bf16(vld1_u16(S3p - 4));
                float32x4_t _S31 = vcvt_f32_bf16(vld1_u16(S3p + 0));
                float32x4_t _S32 = vcvt_f32_bf16(vld1_u16(S3p + 4));
                float32x4_t _S33 = vcvt_f32_bf16(vld1_u16(S3p + 8));
                float32x4_t _rows3 = vmulq_lane_f32(_S30, vget_low_f32(_a0123), 0);
                _rows3 = vmlaq_lane_f32(_rows3, _S31, vget_low_f32(_a0123), 1);
                _rows3 = vmlaq_lane_f32(_rows3, _S32, vget_high_f32(_a0123), 0);
                _rows3 = vmlaq_lane_f32(_rows3, _S33, vget_high_f32(_a0123), 1);
                vst1q_f32(rows3p + dx * 4, _rows3);

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
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float32x4_t _a0123 = vld1q_f32(alphap);

                float32x4_t _S20 = vcvt_f32_bf16(vld1_u16(S2p - 4));
                float32x4_t _S21 = vcvt_f32_bf16(vld1_u16(S2p + 0));
                float32x4_t _S22 = vcvt_f32_bf16(vld1_u16(S2p + 4));
                float32x4_t _S23 = vcvt_f32_bf16(vld1_u16(S2p + 8));
                float32x4_t _S30 = vcvt_f32_bf16(vld1_u16(S3p - 4));
                float32x4_t _S31 = vcvt_f32_bf16(vld1_u16(S3p + 0));
                float32x4_t _S32 = vcvt_f32_bf16(vld1_u16(S3p + 4));
                float32x4_t _S33 = vcvt_f32_bf16(vld1_u16(S3p + 8));
                float32x4_t _rows2 = vmulq_lane_f32(_S20, vget_low_f32(_a0123), 0);
                float32x4_t _rows3 = vmulq_lane_f32(_S30, vget_low_f32(_a0123), 0);
                _rows2 = vmlaq_lane_f32(_rows2, _S21, vget_low_f32(_a0123), 1);
                _rows3 = vmlaq_lane_f32(_rows3, _S31, vget_low_f32(_a0123), 1);
                _rows2 = vmlaq_lane_f32(_rows2, _S22, vget_high_f32(_a0123), 0);
                _rows3 = vmlaq_lane_f32(_rows3, _S32, vget_high_f32(_a0123), 0);
                _rows2 = vmlaq_lane_f32(_rows2, _S23, vget_high_f32(_a0123), 1);
                _rows3 = vmlaq_lane_f32(_rows3, _S33, vget_high_f32(_a0123), 1);
                vst1q_f32(rows2p + dx * 4, _rows2);
                vst1q_f32(rows3p + dx * 4, _rows3);

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
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float32x4_t _a0123 = vld1q_f32(alphap);

                float32x4_t _S10 = vcvt_f32_bf16(vld1_u16(S1p - 4));
                float32x4_t _S11 = vcvt_f32_bf16(vld1_u16(S1p + 0));
                float32x4_t _S12 = vcvt_f32_bf16(vld1_u16(S1p + 4));
                float32x4_t _S13 = vcvt_f32_bf16(vld1_u16(S1p + 8));
                float32x4_t _S20 = vcvt_f32_bf16(vld1_u16(S2p - 4));
                float32x4_t _S21 = vcvt_f32_bf16(vld1_u16(S2p + 0));
                float32x4_t _S22 = vcvt_f32_bf16(vld1_u16(S2p + 4));
                float32x4_t _S23 = vcvt_f32_bf16(vld1_u16(S2p + 8));
                float32x4_t _S30 = vcvt_f32_bf16(vld1_u16(S3p - 4));
                float32x4_t _S31 = vcvt_f32_bf16(vld1_u16(S3p + 0));
                float32x4_t _S32 = vcvt_f32_bf16(vld1_u16(S3p + 4));
                float32x4_t _S33 = vcvt_f32_bf16(vld1_u16(S3p + 8));
                float32x4_t _rows1 = vmulq_lane_f32(_S10, vget_low_f32(_a0123), 0);
                float32x4_t _rows2 = vmulq_lane_f32(_S20, vget_low_f32(_a0123), 0);
                float32x4_t _rows3 = vmulq_lane_f32(_S30, vget_low_f32(_a0123), 0);
                _rows1 = vmlaq_lane_f32(_rows1, _S11, vget_low_f32(_a0123), 1);
                _rows2 = vmlaq_lane_f32(_rows2, _S21, vget_low_f32(_a0123), 1);
                _rows3 = vmlaq_lane_f32(_rows3, _S31, vget_low_f32(_a0123), 1);
                _rows1 = vmlaq_lane_f32(_rows1, _S12, vget_high_f32(_a0123), 0);
                _rows2 = vmlaq_lane_f32(_rows2, _S22, vget_high_f32(_a0123), 0);
                _rows3 = vmlaq_lane_f32(_rows3, _S32, vget_high_f32(_a0123), 0);
                _rows1 = vmlaq_lane_f32(_rows1, _S13, vget_high_f32(_a0123), 1);
                _rows2 = vmlaq_lane_f32(_rows2, _S23, vget_high_f32(_a0123), 1);
                _rows3 = vmlaq_lane_f32(_rows3, _S33, vget_high_f32(_a0123), 1);
                vst1q_f32(rows1p + dx * 4, _rows1);
                vst1q_f32(rows2p + dx * 4, _rows2);
                vst1q_f32(rows3p + dx * 4, _rows3);

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const unsigned short* S0 = src.row<const unsigned short>(sy - 1);
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float32x4_t _a0123 = vld1q_f32(alphap);

                // TODO check the generated assembly on armv7
                float32x4_t _S00 = vcvt_f32_bf16(vld1_u16(S0p - 4));
                float32x4_t _S01 = vcvt_f32_bf16(vld1_u16(S0p + 0));
                float32x4_t _S02 = vcvt_f32_bf16(vld1_u16(S0p + 4));
                float32x4_t _S03 = vcvt_f32_bf16(vld1_u16(S0p + 8));
                float32x4_t _S10 = vcvt_f32_bf16(vld1_u16(S1p - 4));
                float32x4_t _S11 = vcvt_f32_bf16(vld1_u16(S1p + 0));
                float32x4_t _S12 = vcvt_f32_bf16(vld1_u16(S1p + 4));
                float32x4_t _S13 = vcvt_f32_bf16(vld1_u16(S1p + 8));
                float32x4_t _S20 = vcvt_f32_bf16(vld1_u16(S2p - 4));
                float32x4_t _S21 = vcvt_f32_bf16(vld1_u16(S2p + 0));
                float32x4_t _S22 = vcvt_f32_bf16(vld1_u16(S2p + 4));
                float32x4_t _S23 = vcvt_f32_bf16(vld1_u16(S2p + 8));
                float32x4_t _S30 = vcvt_f32_bf16(vld1_u16(S3p - 4));
                float32x4_t _S31 = vcvt_f32_bf16(vld1_u16(S3p + 0));
                float32x4_t _S32 = vcvt_f32_bf16(vld1_u16(S3p + 4));
                float32x4_t _S33 = vcvt_f32_bf16(vld1_u16(S3p + 8));
                float32x4_t _rows0 = vmulq_lane_f32(_S00, vget_low_f32(_a0123), 0);
                float32x4_t _rows1 = vmulq_lane_f32(_S10, vget_low_f32(_a0123), 0);
                float32x4_t _rows2 = vmulq_lane_f32(_S20, vget_low_f32(_a0123), 0);
                float32x4_t _rows3 = vmulq_lane_f32(_S30, vget_low_f32(_a0123), 0);
                _rows0 = vmlaq_lane_f32(_rows0, _S01, vget_low_f32(_a0123), 1);
                _rows1 = vmlaq_lane_f32(_rows1, _S11, vget_low_f32(_a0123), 1);
                _rows2 = vmlaq_lane_f32(_rows2, _S21, vget_low_f32(_a0123), 1);
                _rows3 = vmlaq_lane_f32(_rows3, _S31, vget_low_f32(_a0123), 1);
                _rows0 = vmlaq_lane_f32(_rows0, _S02, vget_high_f32(_a0123), 0);
                _rows1 = vmlaq_lane_f32(_rows1, _S12, vget_high_f32(_a0123), 0);
                _rows2 = vmlaq_lane_f32(_rows2, _S22, vget_high_f32(_a0123), 0);
                _rows3 = vmlaq_lane_f32(_rows3, _S32, vget_high_f32(_a0123), 0);
                _rows0 = vmlaq_lane_f32(_rows0, _S03, vget_high_f32(_a0123), 1);
                _rows1 = vmlaq_lane_f32(_rows1, _S13, vget_high_f32(_a0123), 1);
                _rows2 = vmlaq_lane_f32(_rows2, _S23, vget_high_f32(_a0123), 1);
                _rows3 = vmlaq_lane_f32(_rows3, _S33, vget_high_f32(_a0123), 1);
                vst1q_f32(rows0p + dx * 4, _rows0);
                vst1q_f32(rows1p + dx * 4, _rows1);
                vst1q_f32(rows2p + dx * 4, _rows2);
                vst1q_f32(rows3p + dx * 4, _rows3);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float32x4_t _b0123 = vld1q_f32(beta);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            float32x4_t _rows2 = vld1q_f32(rows2p);
            float32x4_t _rows3 = vld1q_f32(rows3p);
            float32x4_t _D = vmulq_lane_f32(_rows0, vget_low_f32(_b0123), 0);
            _D = vmlaq_lane_f32(_D, _rows1, vget_low_f32(_b0123), 1);
            _D = vmlaq_lane_f32(_D, _rows2, vget_high_f32(_b0123), 0);
            _D = vmlaq_lane_f32(_D, _rows3, vget_high_f32(_b0123), 1);
            vst1_u16(Dp, vcvt_bf16_f32(_D));

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
            rows2p += 4;
            rows3p += 4;
        }

        beta += 4;
    }
}

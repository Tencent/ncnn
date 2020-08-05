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

static void conv3x3s2_pack1to8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + p * 8) : vdupq_n_f16((__fp16)0.f);
        out0.fill(_bias0);

        const __fp16* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            __fp16* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);

            float16x8_t _k00 = vld1q_f16(k0);
            float16x8_t _k01 = vld1q_f16(k0 + 8);
            float16x8_t _k02 = vld1q_f16(k0 + 16);
            float16x8_t _k10 = vld1q_f16(k0 + 24);
            float16x8_t _k11 = vld1q_f16(k0 + 32);
            float16x8_t _k12 = vld1q_f16(k0 + 40);
            float16x8_t _k20 = vld1q_f16(k0 + 48);
            float16x8_t _k21 = vld1q_f16(k0 + 56);
            float16x8_t _k22 = vld1q_f16(k0 + 64);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
                    float16x8_t _sum0 = vld1q_f16(outptr0);
                    float16x8_t _sum1 = vld1q_f16(outptr0 + 8);
                    float16x8_t _sum2 = vld1q_f16(outptr0 + 16);
                    float16x8_t _sum3 = vld1q_f16(outptr0 + 24);

                    float16x8_t _r0 = vld1q_f16(r0);
                    float16x8_t _r1 = vld1q_f16(r1);
                    float16x8_t _r2 = vld1q_f16(r2);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;

                    float16x4_t _r0n = vld1_dup_f16(r0);
                    float16x4_t _r1n = vld1_dup_f16(r1);
                    float16x4_t _r2n = vld1_dup_f16(r2);

                    _sum0 = vfmaq_laneq_f16(_sum0, _k00, _r0, 0);
                    _sum0 = vfmaq_laneq_f16(_sum0, _k01, _r0, 1);
                    _sum0 = vfmaq_laneq_f16(_sum0, _k02, _r0, 2);
                    _sum0 = vfmaq_laneq_f16(_sum0, _k10, _r1, 0);
                    _sum0 = vfmaq_laneq_f16(_sum0, _k11, _r1, 1);
                    _sum0 = vfmaq_laneq_f16(_sum0, _k12, _r1, 2);
                    _sum0 = vfmaq_laneq_f16(_sum0, _k20, _r2, 0);
                    _sum0 = vfmaq_laneq_f16(_sum0, _k21, _r2, 1);
                    _sum0 = vfmaq_laneq_f16(_sum0, _k22, _r2, 2);

                    _sum1 = vfmaq_laneq_f16(_sum1, _k00, _r0, 2);
                    _sum1 = vfmaq_laneq_f16(_sum1, _k01, _r0, 3);
                    _sum1 = vfmaq_laneq_f16(_sum1, _k02, _r0, 4);
                    _sum1 = vfmaq_laneq_f16(_sum1, _k10, _r1, 2);
                    _sum1 = vfmaq_laneq_f16(_sum1, _k11, _r1, 3);
                    _sum1 = vfmaq_laneq_f16(_sum1, _k12, _r1, 4);
                    _sum1 = vfmaq_laneq_f16(_sum1, _k20, _r2, 2);
                    _sum1 = vfmaq_laneq_f16(_sum1, _k21, _r2, 3);
                    _sum1 = vfmaq_laneq_f16(_sum1, _k22, _r2, 4);

                    _sum2 = vfmaq_laneq_f16(_sum2, _k00, _r0, 4);
                    _sum2 = vfmaq_laneq_f16(_sum2, _k01, _r0, 5);
                    _sum2 = vfmaq_laneq_f16(_sum2, _k02, _r0, 6);
                    _sum2 = vfmaq_laneq_f16(_sum2, _k10, _r1, 4);
                    _sum2 = vfmaq_laneq_f16(_sum2, _k11, _r1, 5);
                    _sum2 = vfmaq_laneq_f16(_sum2, _k12, _r1, 6);
                    _sum2 = vfmaq_laneq_f16(_sum2, _k20, _r2, 4);
                    _sum2 = vfmaq_laneq_f16(_sum2, _k21, _r2, 5);
                    _sum2 = vfmaq_laneq_f16(_sum2, _k22, _r2, 6);

                    _sum3 = vfmaq_laneq_f16(_sum3, _k00, _r0, 6);
                    _sum3 = vfmaq_laneq_f16(_sum3, _k01, _r0, 7);
                    _sum3 = vfmaq_lane_f16(_sum3, _k02, _r0n, 0);
                    _sum3 = vfmaq_laneq_f16(_sum3, _k10, _r1, 6);
                    _sum3 = vfmaq_laneq_f16(_sum3, _k11, _r1, 7);
                    _sum3 = vfmaq_lane_f16(_sum3, _k12, _r1n, 0);
                    _sum3 = vfmaq_laneq_f16(_sum3, _k20, _r2, 6);
                    _sum3 = vfmaq_laneq_f16(_sum3, _k21, _r2, 7);
                    _sum3 = vfmaq_lane_f16(_sum3, _k22, _r2n, 0);

                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8, _sum1);
                    vst1q_f16(outptr0 + 16, _sum2);
                    vst1q_f16(outptr0 + 24, _sum3);

                    outptr0 += 32;
                }
                for (; j + 1 < outw; j += 2)
                {
                    float16x8_t _sum0 = vld1q_f16(outptr0);
                    float16x8_t _sum1 = vld1q_f16(outptr0 + 8);

                    float16x4_t _r0 = vld1_f16(r0);
                    float16x4_t _r1 = vld1_f16(r1);
                    float16x4_t _r2 = vld1_f16(r2);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;

                    float16x4_t _r0n = vld1_dup_f16(r0);
                    float16x4_t _r1n = vld1_dup_f16(r1);
                    float16x4_t _r2n = vld1_dup_f16(r2);

                    _sum0 = vfmaq_lane_f16(_sum0, _k00, _r0, 0);
                    _sum0 = vfmaq_lane_f16(_sum0, _k01, _r0, 1);
                    _sum0 = vfmaq_lane_f16(_sum0, _k02, _r0, 2);
                    _sum0 = vfmaq_lane_f16(_sum0, _k10, _r1, 0);
                    _sum0 = vfmaq_lane_f16(_sum0, _k11, _r1, 1);
                    _sum0 = vfmaq_lane_f16(_sum0, _k12, _r1, 2);
                    _sum0 = vfmaq_lane_f16(_sum0, _k20, _r2, 0);
                    _sum0 = vfmaq_lane_f16(_sum0, _k21, _r2, 1);
                    _sum0 = vfmaq_lane_f16(_sum0, _k22, _r2, 2);

                    _sum1 = vfmaq_lane_f16(_sum1, _k00, _r0, 2);
                    _sum1 = vfmaq_lane_f16(_sum1, _k01, _r0, 3);
                    _sum1 = vfmaq_lane_f16(_sum1, _k02, _r0n, 0);
                    _sum1 = vfmaq_lane_f16(_sum1, _k10, _r1, 2);
                    _sum1 = vfmaq_lane_f16(_sum1, _k11, _r1, 3);
                    _sum1 = vfmaq_lane_f16(_sum1, _k12, _r1n, 0);
                    _sum1 = vfmaq_lane_f16(_sum1, _k20, _r2, 2);
                    _sum1 = vfmaq_lane_f16(_sum1, _k21, _r2, 3);
                    _sum1 = vfmaq_lane_f16(_sum1, _k22, _r2n, 0);

                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8, _sum1);

                    outptr0 += 16;
                }
                for (; j < outw; j++)
                {
                    float16x8_t _sum0 = vld1q_f16(outptr0);

                    float16x4_t _r0 = vld1_f16(r0);
                    float16x4_t _r1 = vld1_f16(r1);
                    float16x4_t _r2 = vld1_f16(r2);

                    _sum0 = vfmaq_lane_f16(_sum0, _k00, _r0, 0);
                    _sum0 = vfmaq_lane_f16(_sum0, _k01, _r0, 1);
                    _sum0 = vfmaq_lane_f16(_sum0, _k02, _r0, 2);
                    _sum0 = vfmaq_lane_f16(_sum0, _k10, _r1, 0);
                    _sum0 = vfmaq_lane_f16(_sum0, _k11, _r1, 1);
                    _sum0 = vfmaq_lane_f16(_sum0, _k12, _r1, 2);
                    _sum0 = vfmaq_lane_f16(_sum0, _k20, _r2, 0);
                    _sum0 = vfmaq_lane_f16(_sum0, _k21, _r2, 1);
                    _sum0 = vfmaq_lane_f16(_sum0, _k22, _r2, 2);

                    vst1q_f16(outptr0, _sum0);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 8;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * 8;
        }
    }
}

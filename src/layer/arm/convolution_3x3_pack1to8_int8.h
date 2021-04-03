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

static void conv3x3s1_pack1to8_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        int32x4_t _bias0 = vdupq_n_s32(0);
        out0.fill(_bias0, _bias0);

        const signed char* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            int* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const signed char* r0 = img0.row<const signed char>(0);
            const signed char* r1 = img0.row<const signed char>(1);
            const signed char* r2 = img0.row<const signed char>(2);

            int8x8_t _k00 = vld1_s8(k0);
            int8x8_t _k01 = vld1_s8(k0 + 8);
            int8x8_t _k02 = vld1_s8(k0 + 16);
            int8x8_t _k10 = vld1_s8(k0 + 24);
            int8x8_t _k11 = vld1_s8(k0 + 32);
            int8x8_t _k12 = vld1_s8(k0 + 40);
            int8x8_t _k20 = vld1_s8(k0 + 48);
            int8x8_t _k21 = vld1_s8(k0 + 56);
            int8x8_t _k22 = vld1_s8(k0 + 64);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 1 < outw; j += 2)
                {
                    int32x4_t _sum0 = vld1q_s32(outptr0);
                    int32x4_t _sum1 = vld1q_s32(outptr0 + 4);
                    int32x4_t _sum2 = vld1q_s32(outptr0 + 8);
                    int32x4_t _sum3 = vld1q_s32(outptr0 + 12);

                    int8x8_t _r00 = vld1_dup_s8(r0);
                    int8x8_t _r01 = vld1_dup_s8(r0 + 1);
                    int8x8_t _r02 = vld1_dup_s8(r0 + 2);
                    int8x8_t _r03 = vld1_dup_s8(r0 + 3);

                    int16x8_t _s0_00 = vmull_s8(_r00, _k00);
                    int16x8_t _s1_00 = vmull_s8(_r01, _k00);
                    int16x8_t _s0_01 = vmull_s8(_r01, _k01);
                    int16x8_t _s1_01 = vmull_s8(_r02, _k01);

                    int8x8_t _r10 = vld1_dup_s8(r1);
                    int8x8_t _r11 = vld1_dup_s8(r1 + 1);
                    int8x8_t _r12 = vld1_dup_s8(r1 + 2);
                    int8x8_t _r13 = vld1_dup_s8(r1 + 3);

                    _s0_00 = vmlal_s8(_s0_00, _r02, _k02);
                    _s1_00 = vmlal_s8(_s1_00, _r03, _k02);
                    _s0_01 = vmlal_s8(_s0_01, _r10, _k10);
                    _s1_01 = vmlal_s8(_s1_01, _r11, _k10);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_00));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_00));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_00));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_00));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_01));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_01));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_01));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_01));

                    int16x8_t _s0_11 = vmull_s8(_r11, _k11);
                    int16x8_t _s1_11 = vmull_s8(_r12, _k11);
                    int16x8_t _s0_12 = vmull_s8(_r12, _k12);
                    int16x8_t _s1_12 = vmull_s8(_r13, _k12);

                    int8x8_t _r20 = vld1_dup_s8(r2);
                    int8x8_t _r21 = vld1_dup_s8(r2 + 1);
                    int8x8_t _r22 = vld1_dup_s8(r2 + 2);
                    int8x8_t _r23 = vld1_dup_s8(r2 + 3);

                    _s0_11 = vmlal_s8(_s0_11, _r20, _k20);
                    _s1_11 = vmlal_s8(_s1_11, _r21, _k20);
                    _s0_12 = vmlal_s8(_s0_12, _r21, _k21);
                    _s1_12 = vmlal_s8(_s1_12, _r22, _k21);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_11));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_11));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_11));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_11));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_12));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_12));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_12));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_12));

                    int16x8_t _s0_22 = vmull_s8(_r22, _k22);
                    int16x8_t _s1_22 = vmull_s8(_r23, _k22);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_22));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_22));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_22));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_22));

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 16;
                }
                for (; j < outw; j++)
                {
                    int8x8_t _r00 = vld1_dup_s8(r0);
                    int8x8_t _r01 = vld1_dup_s8(r0 + 1);
                    int8x8_t _r02 = vld1_dup_s8(r0 + 2);
                    int8x8_t _r10 = vld1_dup_s8(r1);
                    int8x8_t _r11 = vld1_dup_s8(r1 + 1);
                    int8x8_t _r12 = vld1_dup_s8(r1 + 2);
                    int8x8_t _r20 = vld1_dup_s8(r2);
                    int8x8_t _r21 = vld1_dup_s8(r2 + 1);
                    int8x8_t _r22 = vld1_dup_s8(r2 + 2);

                    int16x8_t _s00 = vmull_s8(_r00, _k00);
                    int16x8_t _s01 = vmull_s8(_r01, _k01);
                    int16x8_t _s02 = vmull_s8(_r02, _k02);
                    int16x8_t _s10 = vmull_s8(_r10, _k10);

                    _s00 = vmlal_s8(_s00, _r11, _k11);
                    _s01 = vmlal_s8(_s01, _r12, _k12);
                    _s02 = vmlal_s8(_s02, _r20, _k20);
                    _s10 = vmlal_s8(_s10, _r21, _k21);

                    int16x8_t _s22 = vmull_s8(_r22, _k22);

                    int32x4_t _sum0 = vld1q_s32(outptr0);
                    int32x4_t _sum1 = vld1q_s32(outptr0 + 4);
                    int32x4_t _sum2 = vaddl_s16(vget_low_s16(_s00), vget_low_s16(_s01));
                    int32x4_t _sum3 = vaddl_s16(vget_high_s16(_s00), vget_high_s16(_s01));
                    int32x4_t _sum4 = vaddl_s16(vget_low_s16(_s02), vget_low_s16(_s10));
                    int32x4_t _sum5 = vaddl_s16(vget_high_s16(_s02), vget_high_s16(_s10));

                    _sum0 = vaddq_s32(_sum0, _sum2);
                    _sum1 = vaddq_s32(_sum1, _sum3);
                    _sum4 = vaddw_s16(_sum4, vget_low_s16(_s22));
                    _sum5 = vaddw_s16(_sum5, vget_high_s16(_s22));
                    _sum0 = vaddq_s32(_sum0, _sum4);
                    _sum1 = vaddq_s32(_sum1, _sum5);

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                    outptr0 += 8;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * 8;
        }
    }
}

static void conv3x3s2_pack1to8_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        int32x4_t _bias0 = vdupq_n_s32(0);
        out0.fill(_bias0, _bias0);

        const signed char* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            int* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const signed char* r0 = img0.row<const signed char>(0);
            const signed char* r1 = img0.row<const signed char>(1);
            const signed char* r2 = img0.row<const signed char>(2);

            int8x8_t _k00 = vld1_s8(k0);
            int8x8_t _k01 = vld1_s8(k0 + 8);
            int8x8_t _k02 = vld1_s8(k0 + 16);
            int8x8_t _k10 = vld1_s8(k0 + 24);
            int8x8_t _k11 = vld1_s8(k0 + 32);
            int8x8_t _k12 = vld1_s8(k0 + 40);
            int8x8_t _k20 = vld1_s8(k0 + 48);
            int8x8_t _k21 = vld1_s8(k0 + 56);
            int8x8_t _k22 = vld1_s8(k0 + 64);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 1 < outw; j += 2)
                {
                    int32x4_t _sum0 = vld1q_s32(outptr0);
                    int32x4_t _sum1 = vld1q_s32(outptr0 + 4);
                    int32x4_t _sum2 = vld1q_s32(outptr0 + 8);
                    int32x4_t _sum3 = vld1q_s32(outptr0 + 12);

                    int8x8_t _r00 = vld1_dup_s8(r0);
                    int8x8_t _r01 = vld1_dup_s8(r0 + 1);
                    int8x8_t _r02 = vld1_dup_s8(r0 + 2);
                    int8x8_t _r03 = vld1_dup_s8(r0 + 3);
                    int8x8_t _r04 = vld1_dup_s8(r0 + 4);

                    int16x8_t _s0_00 = vmull_s8(_r00, _k00);
                    int16x8_t _s1_00 = vmull_s8(_r02, _k00);
                    int16x8_t _s0_01 = vmull_s8(_r01, _k01);
                    int16x8_t _s1_01 = vmull_s8(_r03, _k01);

                    int8x8_t _r10 = vld1_dup_s8(r1);
                    int8x8_t _r11 = vld1_dup_s8(r1 + 1);
                    int8x8_t _r12 = vld1_dup_s8(r1 + 2);
                    int8x8_t _r13 = vld1_dup_s8(r1 + 3);
                    int8x8_t _r14 = vld1_dup_s8(r1 + 4);

                    _s0_00 = vmlal_s8(_s0_00, _r02, _k02);
                    _s1_00 = vmlal_s8(_s1_00, _r04, _k02);
                    _s0_01 = vmlal_s8(_s0_01, _r10, _k10);
                    _s1_01 = vmlal_s8(_s1_01, _r12, _k10);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_00));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_00));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_00));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_00));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_01));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_01));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_01));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_01));

                    int16x8_t _s0_11 = vmull_s8(_r11, _k11);
                    int16x8_t _s1_11 = vmull_s8(_r13, _k11);
                    int16x8_t _s0_12 = vmull_s8(_r12, _k12);
                    int16x8_t _s1_12 = vmull_s8(_r14, _k12);

                    int8x8_t _r20 = vld1_dup_s8(r2);
                    int8x8_t _r21 = vld1_dup_s8(r2 + 1);
                    int8x8_t _r22 = vld1_dup_s8(r2 + 2);
                    int8x8_t _r23 = vld1_dup_s8(r2 + 3);
                    int8x8_t _r24 = vld1_dup_s8(r2 + 4);

                    _s0_11 = vmlal_s8(_s0_11, _r20, _k20);
                    _s1_11 = vmlal_s8(_s1_11, _r22, _k20);
                    _s0_12 = vmlal_s8(_s0_12, _r21, _k21);
                    _s1_12 = vmlal_s8(_s1_12, _r23, _k21);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_11));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_11));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_11));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_11));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_12));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_12));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_12));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_12));

                    int16x8_t _s0_22 = vmull_s8(_r22, _k22);
                    int16x8_t _s1_22 = vmull_s8(_r24, _k22);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0_22));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0_22));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1_22));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1_22));

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr0 += 16;
                }
                for (; j < outw; j++)
                {
                    int8x8_t _r00 = vld1_dup_s8(r0);
                    int8x8_t _r01 = vld1_dup_s8(r0 + 1);
                    int8x8_t _r02 = vld1_dup_s8(r0 + 2);
                    int8x8_t _r10 = vld1_dup_s8(r1);
                    int8x8_t _r11 = vld1_dup_s8(r1 + 1);
                    int8x8_t _r12 = vld1_dup_s8(r1 + 2);
                    int8x8_t _r20 = vld1_dup_s8(r2);
                    int8x8_t _r21 = vld1_dup_s8(r2 + 1);
                    int8x8_t _r22 = vld1_dup_s8(r2 + 2);

                    int16x8_t _s00 = vmull_s8(_r00, _k00);
                    int16x8_t _s01 = vmull_s8(_r01, _k01);
                    int16x8_t _s02 = vmull_s8(_r02, _k02);
                    int16x8_t _s10 = vmull_s8(_r10, _k10);

                    _s00 = vmlal_s8(_s00, _r11, _k11);
                    _s01 = vmlal_s8(_s01, _r12, _k12);
                    _s02 = vmlal_s8(_s02, _r20, _k20);
                    _s10 = vmlal_s8(_s10, _r21, _k21);

                    int16x8_t _s22 = vmull_s8(_r22, _k22);

                    int32x4_t _sum0 = vld1q_s32(outptr0);
                    int32x4_t _sum1 = vld1q_s32(outptr0 + 4);
                    int32x4_t _sum2 = vaddl_s16(vget_low_s16(_s00), vget_low_s16(_s01));
                    int32x4_t _sum3 = vaddl_s16(vget_high_s16(_s00), vget_high_s16(_s01));
                    int32x4_t _sum4 = vaddl_s16(vget_low_s16(_s02), vget_low_s16(_s10));
                    int32x4_t _sum5 = vaddl_s16(vget_high_s16(_s02), vget_high_s16(_s10));

                    _sum0 = vaddq_s32(_sum0, _sum2);
                    _sum1 = vaddq_s32(_sum1, _sum3);
                    _sum4 = vaddw_s16(_sum4, vget_low_s16(_s22));
                    _sum5 = vaddw_s16(_sum5, vget_high_s16(_s22));
                    _sum0 = vaddq_s32(_sum0, _sum4);
                    _sum1 = vaddq_s32(_sum1, _sum5);

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);

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

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

static void conv3x3s1_pack1to4_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        v4f32 _bias0 = bias ? (v4f32)__lsx_vld(bias + p * 4, 0) : (v4f32)__lsx_vreplgr2vr_w(0);
        out0.fill(_bias0);

        const float* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            float* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            v4f32 _k00 = (v4f32)__lsx_vld(k0, 0);
            v4f32 _k01 = (v4f32)__lsx_vld(k0 + 4, 0);
            v4f32 _k02 = (v4f32)__lsx_vld(k0 + 4 * 2, 0);
            v4f32 _k10 = (v4f32)__lsx_vld(k0 + 4 * 3, 0);
            v4f32 _k11 = (v4f32)__lsx_vld(k0 + 4 * 4, 0);
            v4f32 _k12 = (v4f32)__lsx_vld(k0 + 4 * 5, 0);
            v4f32 _k20 = (v4f32)__lsx_vld(k0 + 4 * 6, 0);
            v4f32 _k21 = (v4f32)__lsx_vld(k0 + 4 * 7, 0);
            v4f32 _k22 = (v4f32)__lsx_vld(k0 + 4 * 8, 0);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    v4f32 _sum0 = (v4f32)__lsx_vld(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__lsx_vld(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__lsx_vld(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__lsx_vld(outptr0 + 4 * 3, 0);
                    v4f32 _sum4 = (v4f32)__lsx_vld(outptr0 + 4 * 4, 0);
                    v4f32 _sum5 = (v4f32)__lsx_vld(outptr0 + 4 * 5, 0);
                    v4f32 _sum6 = (v4f32)__lsx_vld(outptr0 + 4 * 6, 0);
                    v4f32 _sum7 = (v4f32)__lsx_vld(outptr0 + 4 * 7, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);
                    __m128i _r0nn = __lsx_vld(r0 + 8, 0);

                    v4f32 _r00 = (v4f32)__lsx_vreplvei_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__lsx_vreplvei_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__lsx_vreplvei_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__lsx_vreplvei_w(_r0, 3);
                    v4f32 _r04 = (v4f32)__lsx_vreplvei_w(_r0n, 0);
                    v4f32 _r05 = (v4f32)__lsx_vreplvei_w(_r0n, 1);
                    v4f32 _r06 = (v4f32)__lsx_vreplvei_w(_r0n, 2);
                    v4f32 _r07 = (v4f32)__lsx_vreplvei_w(_r0n, 3);
                    v4f32 _r08 = (v4f32)__lsx_vreplvei_w(_r0nn, 0);
                    v4f32 _r09 = (v4f32)__lsx_vreplvei_w(_r0nn, 1);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r01, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k00, _r02, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k00, _r03, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k00, _r04, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k00, _r05, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k00, _r06, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k00, _r07, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r02, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k01, _r03, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k01, _r04, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k01, _r05, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k01, _r06, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k01, _r07, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k01, _r08, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r03, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k02, _r04, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k02, _r05, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k02, _r06, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k02, _r07, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k02, _r08, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k02, _r09, _sum7);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r1n = __lsx_vld(r1 + 4, 0);
                    __m128i _r1nn = __lsx_vld(r1 + 8, 0);

                    v4f32 _r10 = (v4f32)__lsx_vreplvei_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__lsx_vreplvei_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__lsx_vreplvei_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__lsx_vreplvei_w(_r1, 3);
                    v4f32 _r14 = (v4f32)__lsx_vreplvei_w(_r1n, 0);
                    v4f32 _r15 = (v4f32)__lsx_vreplvei_w(_r1n, 1);
                    v4f32 _r16 = (v4f32)__lsx_vreplvei_w(_r1n, 2);
                    v4f32 _r17 = (v4f32)__lsx_vreplvei_w(_r1n, 3);
                    v4f32 _r18 = (v4f32)__lsx_vreplvei_w(_r1nn, 0);
                    v4f32 _r19 = (v4f32)__lsx_vreplvei_w(_r1nn, 1);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r11, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k10, _r12, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k10, _r13, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k10, _r14, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k10, _r15, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k10, _r16, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k10, _r17, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r12, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k11, _r13, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k11, _r14, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k11, _r15, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k11, _r16, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k11, _r17, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k11, _r18, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r13, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k12, _r14, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k12, _r15, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k12, _r16, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k12, _r17, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k12, _r18, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k12, _r19, _sum7);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r2n = __lsx_vld(r2 + 4, 0);
                    __m128i _r2nn = __lsx_vld(r2 + 8, 0);

                    v4f32 _r20 = (v4f32)__lsx_vreplvei_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__lsx_vreplvei_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__lsx_vreplvei_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__lsx_vreplvei_w(_r2, 3);
                    v4f32 _r24 = (v4f32)__lsx_vreplvei_w(_r2n, 0);
                    v4f32 _r25 = (v4f32)__lsx_vreplvei_w(_r2n, 1);
                    v4f32 _r26 = (v4f32)__lsx_vreplvei_w(_r2n, 2);
                    v4f32 _r27 = (v4f32)__lsx_vreplvei_w(_r2n, 3);
                    v4f32 _r28 = (v4f32)__lsx_vreplvei_w(_r2nn, 0);
                    v4f32 _r29 = (v4f32)__lsx_vreplvei_w(_r2nn, 1);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k20, _r21, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k20, _r22, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k20, _r23, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k20, _r24, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k20, _r25, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k20, _r26, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k20, _r27, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k21, _r22, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k21, _r23, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k21, _r24, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k21, _r25, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k21, _r26, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k21, _r27, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k21, _r28, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k22, _r23, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k22, _r24, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k22, _r25, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k22, _r26, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k22, _r27, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k22, _r28, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k22, _r29, _sum7);

                    __lsx_vst(_sum0, outptr0, 0);
                    __lsx_vst(_sum1, outptr0 + 4, 0);
                    __lsx_vst(_sum2, outptr0 + 4 * 2, 0);
                    __lsx_vst(_sum3, outptr0 + 4 * 3, 0);
                    __lsx_vst(_sum4, outptr0 + 4 * 4, 0);
                    __lsx_vst(_sum5, outptr0 + 4 * 5, 0);
                    __lsx_vst(_sum6, outptr0 + 4 * 6, 0);
                    __lsx_vst(_sum7, outptr0 + 4 * 7, 0);

                    outptr0 += 4 * 8;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
                for (; j + 3 < outw; j += 4)
                {
                    v4f32 _sum0 = (v4f32)__lsx_vld(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__lsx_vld(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__lsx_vld(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__lsx_vld(outptr0 + 4 * 3, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);

                    v4f32 _r00 = (v4f32)__lsx_vreplvei_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__lsx_vreplvei_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__lsx_vreplvei_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__lsx_vreplvei_w(_r0, 3);
                    v4f32 _r04 = (v4f32)__lsx_vreplvei_w(_r0n, 0);
                    v4f32 _r05 = (v4f32)__lsx_vreplvei_w(_r0n, 1);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r01, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k00, _r02, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k00, _r03, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r02, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k01, _r03, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k01, _r04, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r03, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k02, _r04, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k02, _r05, _sum3);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r1n = __lsx_vld(r1 + 4, 0);

                    v4f32 _r10 = (v4f32)__lsx_vreplvei_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__lsx_vreplvei_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__lsx_vreplvei_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__lsx_vreplvei_w(_r1, 3);
                    v4f32 _r14 = (v4f32)__lsx_vreplvei_w(_r1n, 0);
                    v4f32 _r15 = (v4f32)__lsx_vreplvei_w(_r1n, 1);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r11, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k10, _r12, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k10, _r13, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r12, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k11, _r13, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k11, _r14, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r13, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k12, _r14, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k12, _r15, _sum3);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r2n = __lsx_vld(r2 + 4, 0);

                    v4f32 _r20 = (v4f32)__lsx_vreplvei_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__lsx_vreplvei_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__lsx_vreplvei_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__lsx_vreplvei_w(_r2, 3);
                    v4f32 _r24 = (v4f32)__lsx_vreplvei_w(_r2n, 0);
                    v4f32 _r25 = (v4f32)__lsx_vreplvei_w(_r2n, 1);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k20, _r21, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k20, _r22, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k20, _r23, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k21, _r22, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k21, _r23, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k21, _r24, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k22, _r23, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k22, _r24, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k22, _r25, _sum3);

                    __lsx_vst(_sum0, outptr0, 0);
                    __lsx_vst(_sum1, outptr0 + 4, 0);
                    __lsx_vst(_sum2, outptr0 + 4 * 2, 0);
                    __lsx_vst(_sum3, outptr0 + 4 * 3, 0);

                    outptr0 += 4 * 4;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                for (; j + 1 < outw; j += 2)
                {
                    v4f32 _sum0 = (v4f32)__lsx_vld(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__lsx_vld(outptr0 + 4, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    v4f32 _r00 = (v4f32)__lsx_vreplvei_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__lsx_vreplvei_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__lsx_vreplvei_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__lsx_vreplvei_w(_r0, 3);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r01, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r02, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r03, _sum1);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    v4f32 _r10 = (v4f32)__lsx_vreplvei_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__lsx_vreplvei_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__lsx_vreplvei_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__lsx_vreplvei_w(_r1, 3);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r11, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r12, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r13, _sum1);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    v4f32 _r20 = (v4f32)__lsx_vreplvei_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__lsx_vreplvei_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__lsx_vreplvei_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__lsx_vreplvei_w(_r2, 3);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k20, _r21, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k21, _r22, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k22, _r23, _sum1);

                    __lsx_vst(_sum0, outptr0, 0);
                    __lsx_vst(_sum1, outptr0 + 4, 0);

                    outptr0 += 4 * 2;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }
                for (; j < outw; j++)
                {
                    v4f32 _sum0 = (v4f32)__lsx_vld(outptr0, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    v4f32 _r00 = (v4f32)__lsx_vreplvei_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__lsx_vreplvei_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__lsx_vreplvei_w(_r0, 2);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    v4f32 _r10 = (v4f32)__lsx_vreplvei_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__lsx_vreplvei_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__lsx_vreplvei_w(_r1, 2);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    v4f32 _r20 = (v4f32)__lsx_vreplvei_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__lsx_vreplvei_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__lsx_vreplvei_w(_r2, 2);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);

                    __lsx_vst(_sum0, outptr0, 0);

                    outptr0 += 4;

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * 4;
        }
    }
}

static void conv3x3s2_pack1to4_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        v4f32 _bias0 = bias ? (v4f32)__lsx_vld(bias + p * 4, 0) : (v4f32)__lsx_vreplgr2vr_w(0);
        out0.fill(_bias0);

        const float* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            float* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            v4f32 _k00 = (v4f32)__lsx_vld(k0, 0);
            v4f32 _k01 = (v4f32)__lsx_vld(k0 + 4, 0);
            v4f32 _k02 = (v4f32)__lsx_vld(k0 + 4 * 2, 0);
            v4f32 _k10 = (v4f32)__lsx_vld(k0 + 4 * 3, 0);
            v4f32 _k11 = (v4f32)__lsx_vld(k0 + 4 * 4, 0);
            v4f32 _k12 = (v4f32)__lsx_vld(k0 + 4 * 5, 0);
            v4f32 _k20 = (v4f32)__lsx_vld(k0 + 4 * 6, 0);
            v4f32 _k21 = (v4f32)__lsx_vld(k0 + 4 * 7, 0);
            v4f32 _k22 = (v4f32)__lsx_vld(k0 + 4 * 8, 0);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    v4f32 _sum0 = (v4f32)__lsx_vld(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__lsx_vld(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__lsx_vld(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__lsx_vld(outptr0 + 4 * 3, 0);
                    v4f32 _sum4 = (v4f32)__lsx_vld(outptr0 + 4 * 4, 0);
                    v4f32 _sum5 = (v4f32)__lsx_vld(outptr0 + 4 * 5, 0);
                    v4f32 _sum6 = (v4f32)__lsx_vld(outptr0 + 4 * 6, 0);
                    v4f32 _sum7 = (v4f32)__lsx_vld(outptr0 + 4 * 7, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);
                    __m128i _r0nn = __lsx_vld(r0 + 8, 0);
                    __m128i _r0nnn = __lsx_vld(r0 + 12, 0);

                    v4f32 _r00 = (v4f32)__lsx_vreplvei_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__lsx_vreplvei_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__lsx_vreplvei_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__lsx_vreplvei_w(_r0, 3);
                    v4f32 _r04 = (v4f32)__lsx_vreplvei_w(_r0n, 0);
                    v4f32 _r05 = (v4f32)__lsx_vreplvei_w(_r0n, 1);
                    v4f32 _r06 = (v4f32)__lsx_vreplvei_w(_r0n, 2);
                    v4f32 _r07 = (v4f32)__lsx_vreplvei_w(_r0n, 3);
                    v4f32 _r08 = (v4f32)__lsx_vreplvei_w(_r0nn, 0);
                    v4f32 _r09 = (v4f32)__lsx_vreplvei_w(_r0nn, 1);
                    v4f32 _r0a = (v4f32)__lsx_vreplvei_w(_r0nn, 2);
                    v4f32 _r0b = (v4f32)__lsx_vreplvei_w(_r0nn, 3);
                    v4f32 _r0c = (v4f32)__lsx_vreplvei_w(_r0nnn, 0);
                    v4f32 _r0d = (v4f32)__lsx_vreplvei_w(_r0nnn, 1);
                    v4f32 _r0e = (v4f32)__lsx_vreplvei_w(_r0nnn, 2);
                    v4f32 _r0f = (v4f32)__lsx_vreplvei_w(_r0nnn, 3);
                    v4f32 _r0g = __lsx_vreplfr2vr_s(r0[16]);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r02, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k00, _r04, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k00, _r06, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k00, _r08, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k00, _r0a, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k00, _r0c, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k00, _r0e, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r03, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k01, _r05, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k01, _r07, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k01, _r09, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k01, _r0b, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k01, _r0d, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k01, _r0f, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r04, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k02, _r06, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k02, _r08, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k02, _r0a, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k02, _r0c, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k02, _r0e, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k02, _r0g, _sum7);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r1n = __lsx_vld(r1 + 4, 0);
                    __m128i _r1nn = __lsx_vld(r1 + 8, 0);
                    __m128i _r1nnn = __lsx_vld(r1 + 12, 0);

                    v4f32 _r10 = (v4f32)__lsx_vreplvei_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__lsx_vreplvei_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__lsx_vreplvei_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__lsx_vreplvei_w(_r1, 3);
                    v4f32 _r14 = (v4f32)__lsx_vreplvei_w(_r1n, 0);
                    v4f32 _r15 = (v4f32)__lsx_vreplvei_w(_r1n, 1);
                    v4f32 _r16 = (v4f32)__lsx_vreplvei_w(_r1n, 2);
                    v4f32 _r17 = (v4f32)__lsx_vreplvei_w(_r1n, 3);
                    v4f32 _r18 = (v4f32)__lsx_vreplvei_w(_r1nn, 0);
                    v4f32 _r19 = (v4f32)__lsx_vreplvei_w(_r1nn, 1);
                    v4f32 _r1a = (v4f32)__lsx_vreplvei_w(_r1nn, 2);
                    v4f32 _r1b = (v4f32)__lsx_vreplvei_w(_r1nn, 3);
                    v4f32 _r1c = (v4f32)__lsx_vreplvei_w(_r1nnn, 0);
                    v4f32 _r1d = (v4f32)__lsx_vreplvei_w(_r1nnn, 1);
                    v4f32 _r1e = (v4f32)__lsx_vreplvei_w(_r1nnn, 2);
                    v4f32 _r1f = (v4f32)__lsx_vreplvei_w(_r1nnn, 3);
                    v4f32 _r1g = __lsx_vreplfr2vr_s(r1[16]);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r12, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k10, _r14, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k10, _r16, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k10, _r18, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k10, _r1a, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k10, _r1c, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k10, _r1e, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r13, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k11, _r15, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k11, _r17, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k11, _r19, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k11, _r1b, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k11, _r1d, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k11, _r1f, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r14, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k12, _r16, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k12, _r18, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k12, _r1a, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k12, _r1c, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k12, _r1e, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k12, _r1g, _sum7);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r2n = __lsx_vld(r2 + 4, 0);
                    __m128i _r2nn = __lsx_vld(r2 + 8, 0);
                    __m128i _r2nnn = __lsx_vld(r2 + 12, 0);

                    v4f32 _r20 = (v4f32)__lsx_vreplvei_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__lsx_vreplvei_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__lsx_vreplvei_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__lsx_vreplvei_w(_r2, 3);
                    v4f32 _r24 = (v4f32)__lsx_vreplvei_w(_r2n, 0);
                    v4f32 _r25 = (v4f32)__lsx_vreplvei_w(_r2n, 1);
                    v4f32 _r26 = (v4f32)__lsx_vreplvei_w(_r2n, 2);
                    v4f32 _r27 = (v4f32)__lsx_vreplvei_w(_r2n, 3);
                    v4f32 _r28 = (v4f32)__lsx_vreplvei_w(_r2nn, 0);
                    v4f32 _r29 = (v4f32)__lsx_vreplvei_w(_r2nn, 1);
                    v4f32 _r2a = (v4f32)__lsx_vreplvei_w(_r2nn, 2);
                    v4f32 _r2b = (v4f32)__lsx_vreplvei_w(_r2nn, 3);
                    v4f32 _r2c = (v4f32)__lsx_vreplvei_w(_r2nnn, 0);
                    v4f32 _r2d = (v4f32)__lsx_vreplvei_w(_r2nnn, 1);
                    v4f32 _r2e = (v4f32)__lsx_vreplvei_w(_r2nnn, 2);
                    v4f32 _r2f = (v4f32)__lsx_vreplvei_w(_r2nnn, 3);
                    v4f32 _r2g = __lsx_vreplfr2vr_s(r2[16]);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k20, _r22, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k20, _r24, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k20, _r26, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k20, _r28, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k20, _r2a, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k20, _r2c, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k20, _r2e, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k21, _r23, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k21, _r25, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k21, _r27, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k21, _r29, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k21, _r2b, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k21, _r2d, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k21, _r2f, _sum7);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k22, _r24, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k22, _r26, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k22, _r28, _sum3);
                    _sum4 = __lsx_vfmadd_s(_k22, _r2a, _sum4);
                    _sum5 = __lsx_vfmadd_s(_k22, _r2c, _sum5);
                    _sum6 = __lsx_vfmadd_s(_k22, _r2e, _sum6);
                    _sum7 = __lsx_vfmadd_s(_k22, _r2g, _sum7);

                    __lsx_vst(_sum0, outptr0, 0);
                    __lsx_vst(_sum1, outptr0 + 4, 0);
                    __lsx_vst(_sum2, outptr0 + 4 * 2, 0);
                    __lsx_vst(_sum3, outptr0 + 4 * 3, 0);
                    __lsx_vst(_sum4, outptr0 + 4 * 4, 0);
                    __lsx_vst(_sum5, outptr0 + 4 * 5, 0);
                    __lsx_vst(_sum6, outptr0 + 4 * 6, 0);
                    __lsx_vst(_sum7, outptr0 + 4 * 7, 0);

                    outptr0 += 4 * 8;

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                }
                for (; j + 3 < outw; j += 4)
                {
                    v4f32 _sum0 = (v4f32)__lsx_vld(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__lsx_vld(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__lsx_vld(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__lsx_vld(outptr0 + 4 * 3, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);

                    v4f32 _r00 = (v4f32)__lsx_vreplvei_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__lsx_vreplvei_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__lsx_vreplvei_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__lsx_vreplvei_w(_r0, 3);
                    v4f32 _r04 = (v4f32)__lsx_vreplvei_w(_r0n, 0);
                    v4f32 _r05 = (v4f32)__lsx_vreplvei_w(_r0n, 1);
                    v4f32 _r06 = (v4f32)__lsx_vreplvei_w(_r0n, 2);
                    v4f32 _r07 = (v4f32)__lsx_vreplvei_w(_r0n, 3);
                    v4f32 _r08 = __lsx_vreplfr2vr_s(r0[8]);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r02, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k00, _r04, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k00, _r06, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r03, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k01, _r05, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k01, _r07, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r04, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k02, _r06, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k02, _r08, _sum3);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r1n = __lsx_vld(r1 + 4, 0);

                    v4f32 _r10 = (v4f32)__lsx_vreplvei_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__lsx_vreplvei_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__lsx_vreplvei_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__lsx_vreplvei_w(_r1, 3);
                    v4f32 _r14 = (v4f32)__lsx_vreplvei_w(_r1n, 0);
                    v4f32 _r15 = (v4f32)__lsx_vreplvei_w(_r1n, 1);
                    v4f32 _r16 = (v4f32)__lsx_vreplvei_w(_r1n, 2);
                    v4f32 _r17 = (v4f32)__lsx_vreplvei_w(_r1n, 3);
                    v4f32 _r18 = __lsx_vreplfr2vr_s(r1[8]);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r12, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k10, _r14, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k10, _r16, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r13, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k11, _r15, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k11, _r17, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r14, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k12, _r16, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k12, _r18, _sum3);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r2n = __lsx_vld(r2 + 4, 0);

                    v4f32 _r20 = (v4f32)__lsx_vreplvei_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__lsx_vreplvei_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__lsx_vreplvei_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__lsx_vreplvei_w(_r2, 3);
                    v4f32 _r24 = (v4f32)__lsx_vreplvei_w(_r2n, 0);
                    v4f32 _r25 = (v4f32)__lsx_vreplvei_w(_r2n, 1);
                    v4f32 _r26 = (v4f32)__lsx_vreplvei_w(_r2n, 2);
                    v4f32 _r27 = (v4f32)__lsx_vreplvei_w(_r2n, 3);
                    v4f32 _r28 = __lsx_vreplfr2vr_s(r2[8]);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k20, _r22, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k20, _r24, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k20, _r26, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k21, _r23, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k21, _r25, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k21, _r27, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k22, _r24, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k22, _r26, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k22, _r28, _sum3);

                    __lsx_vst(_sum0, outptr0, 0);
                    __lsx_vst(_sum1, outptr0 + 4, 0);
                    __lsx_vst(_sum2, outptr0 + 4 * 2, 0);
                    __lsx_vst(_sum3, outptr0 + 4 * 3, 0);

                    outptr0 += 4 * 4;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
                for (; j + 1 < outw; j += 2)
                {
                    v4f32 _sum0 = (v4f32)__lsx_vld(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__lsx_vld(outptr0 + 4, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    v4f32 _r00 = (v4f32)__lsx_vreplvei_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__lsx_vreplvei_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__lsx_vreplvei_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__lsx_vreplvei_w(_r0, 3);
                    v4f32 _r04 = __lsx_vreplfr2vr_s(r0[4]);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r02, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r03, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r04, _sum1);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    v4f32 _r10 = (v4f32)__lsx_vreplvei_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__lsx_vreplvei_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__lsx_vreplvei_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__lsx_vreplvei_w(_r1, 3);
                    v4f32 _r14 = __lsx_vreplfr2vr_s(r1[4]);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r12, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r13, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r14, _sum1);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    v4f32 _r20 = (v4f32)__lsx_vreplvei_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__lsx_vreplvei_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__lsx_vreplvei_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__lsx_vreplvei_w(_r2, 3);
                    v4f32 _r24 = __lsx_vreplfr2vr_s(r2[4]);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k20, _r22, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k21, _r23, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k22, _r24, _sum1);

                    __lsx_vst(_sum0, outptr0, 0);
                    __lsx_vst(_sum1, outptr0 + 4, 0);

                    outptr0 += 4 * 2;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                for (; j < outw; j++)
                {
                    v4f32 _sum0 = (v4f32)__lsx_vld(outptr0, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    v4f32 _r00 = (v4f32)__lsx_vreplvei_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__lsx_vreplvei_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__lsx_vreplvei_w(_r0, 2);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    v4f32 _r10 = (v4f32)__lsx_vreplvei_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__lsx_vreplvei_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__lsx_vreplvei_w(_r1, 2);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    v4f32 _r20 = (v4f32)__lsx_vreplvei_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__lsx_vreplvei_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__lsx_vreplvei_w(_r2, 2);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);

                    __lsx_vst(_sum0, outptr0, 0);

                    outptr0 += 4;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * 4;
        }
    }
}

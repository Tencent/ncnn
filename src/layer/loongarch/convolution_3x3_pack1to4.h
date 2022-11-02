// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
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

        __m128 _bias0 = bias ? (__m128)__lsx_vld(bias + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);
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

            __m128 _k00 = (__m128)__lsx_vld(k0, 0);
            __m128 _k01 = (__m128)__lsx_vld(k0 + 4, 0);
            __m128 _k02 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
            __m128 _k10 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
            __m128 _k11 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
            __m128 _k12 = (__m128)__lsx_vld(k0 + 4 * 5, 0);
            __m128 _k20 = (__m128)__lsx_vld(k0 + 4 * 6, 0);
            __m128 _k21 = (__m128)__lsx_vld(k0 + 4 * 7, 0);
            __m128 _k22 = (__m128)__lsx_vld(k0 + 4 * 8, 0);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    __m128 _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);
                    __m128 _sum2 = (__m128)__lsx_vld(outptr0 + 4 * 2, 0);
                    __m128 _sum3 = (__m128)__lsx_vld(outptr0 + 4 * 3, 0);
                    __m128 _sum4 = (__m128)__lsx_vld(outptr0 + 4 * 4, 0);
                    __m128 _sum5 = (__m128)__lsx_vld(outptr0 + 4 * 5, 0);
                    __m128 _sum6 = (__m128)__lsx_vld(outptr0 + 4 * 6, 0);
                    __m128 _sum7 = (__m128)__lsx_vld(outptr0 + 4 * 7, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);
                    __m128i _r0nn = __lsx_vld(r0 + 8, 0);

                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);
                    __m128 _r03 = (__m128)__lsx_vreplvei_w(_r0, 3);
                    __m128 _r04 = (__m128)__lsx_vreplvei_w(_r0n, 0);
                    __m128 _r05 = (__m128)__lsx_vreplvei_w(_r0n, 1);
                    __m128 _r06 = (__m128)__lsx_vreplvei_w(_r0n, 2);
                    __m128 _r07 = (__m128)__lsx_vreplvei_w(_r0n, 3);
                    __m128 _r08 = (__m128)__lsx_vreplvei_w(_r0nn, 0);
                    __m128 _r09 = (__m128)__lsx_vreplvei_w(_r0nn, 1);

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

                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);
                    __m128 _r13 = (__m128)__lsx_vreplvei_w(_r1, 3);
                    __m128 _r14 = (__m128)__lsx_vreplvei_w(_r1n, 0);
                    __m128 _r15 = (__m128)__lsx_vreplvei_w(_r1n, 1);
                    __m128 _r16 = (__m128)__lsx_vreplvei_w(_r1n, 2);
                    __m128 _r17 = (__m128)__lsx_vreplvei_w(_r1n, 3);
                    __m128 _r18 = (__m128)__lsx_vreplvei_w(_r1nn, 0);
                    __m128 _r19 = (__m128)__lsx_vreplvei_w(_r1nn, 1);

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

                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);
                    __m128 _r23 = (__m128)__lsx_vreplvei_w(_r2, 3);
                    __m128 _r24 = (__m128)__lsx_vreplvei_w(_r2n, 0);
                    __m128 _r25 = (__m128)__lsx_vreplvei_w(_r2n, 1);
                    __m128 _r26 = (__m128)__lsx_vreplvei_w(_r2n, 2);
                    __m128 _r27 = (__m128)__lsx_vreplvei_w(_r2n, 3);
                    __m128 _r28 = (__m128)__lsx_vreplvei_w(_r2nn, 0);
                    __m128 _r29 = (__m128)__lsx_vreplvei_w(_r2nn, 1);

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
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    __m128 _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);
                    __m128 _sum2 = (__m128)__lsx_vld(outptr0 + 4 * 2, 0);
                    __m128 _sum3 = (__m128)__lsx_vld(outptr0 + 4 * 3, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);

                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);
                    __m128 _r03 = (__m128)__lsx_vreplvei_w(_r0, 3);
                    __m128 _r04 = (__m128)__lsx_vreplvei_w(_r0n, 0);
                    __m128 _r05 = (__m128)__lsx_vreplvei_w(_r0n, 1);

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

                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);
                    __m128 _r13 = (__m128)__lsx_vreplvei_w(_r1, 3);
                    __m128 _r14 = (__m128)__lsx_vreplvei_w(_r1n, 0);
                    __m128 _r15 = (__m128)__lsx_vreplvei_w(_r1n, 1);

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

                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);
                    __m128 _r23 = (__m128)__lsx_vreplvei_w(_r2, 3);
                    __m128 _r24 = (__m128)__lsx_vreplvei_w(_r2n, 0);
                    __m128 _r25 = (__m128)__lsx_vreplvei_w(_r2n, 1);

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
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    __m128 _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);
                    __m128 _r03 = (__m128)__lsx_vreplvei_w(_r0, 3);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r01, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r02, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r03, _sum1);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);
                    __m128 _r13 = (__m128)__lsx_vreplvei_w(_r1, 3);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r11, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r12, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r13, _sum1);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);
                    __m128 _r23 = (__m128)__lsx_vreplvei_w(_r2, 3);

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
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);

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

        __m128 _bias0 = bias ? (__m128)__lsx_vld(bias + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);
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

            __m128 _k00 = (__m128)__lsx_vld(k0, 0);
            __m128 _k01 = (__m128)__lsx_vld(k0 + 4, 0);
            __m128 _k02 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
            __m128 _k10 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
            __m128 _k11 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
            __m128 _k12 = (__m128)__lsx_vld(k0 + 4 * 5, 0);
            __m128 _k20 = (__m128)__lsx_vld(k0 + 4 * 6, 0);
            __m128 _k21 = (__m128)__lsx_vld(k0 + 4 * 7, 0);
            __m128 _k22 = (__m128)__lsx_vld(k0 + 4 * 8, 0);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    __m128 _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);
                    __m128 _sum2 = (__m128)__lsx_vld(outptr0 + 4 * 2, 0);
                    __m128 _sum3 = (__m128)__lsx_vld(outptr0 + 4 * 3, 0);
                    __m128 _sum4 = (__m128)__lsx_vld(outptr0 + 4 * 4, 0);
                    __m128 _sum5 = (__m128)__lsx_vld(outptr0 + 4 * 5, 0);
                    __m128 _sum6 = (__m128)__lsx_vld(outptr0 + 4 * 6, 0);
                    __m128 _sum7 = (__m128)__lsx_vld(outptr0 + 4 * 7, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);
                    __m128i _r0nn = __lsx_vld(r0 + 8, 0);
                    __m128i _r0nnn = __lsx_vld(r0 + 12, 0);

                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);
                    __m128 _r03 = (__m128)__lsx_vreplvei_w(_r0, 3);
                    __m128 _r04 = (__m128)__lsx_vreplvei_w(_r0n, 0);
                    __m128 _r05 = (__m128)__lsx_vreplvei_w(_r0n, 1);
                    __m128 _r06 = (__m128)__lsx_vreplvei_w(_r0n, 2);
                    __m128 _r07 = (__m128)__lsx_vreplvei_w(_r0n, 3);
                    __m128 _r08 = (__m128)__lsx_vreplvei_w(_r0nn, 0);
                    __m128 _r09 = (__m128)__lsx_vreplvei_w(_r0nn, 1);
                    __m128 _r0a = (__m128)__lsx_vreplvei_w(_r0nn, 2);
                    __m128 _r0b = (__m128)__lsx_vreplvei_w(_r0nn, 3);
                    __m128 _r0c = (__m128)__lsx_vreplvei_w(_r0nnn, 0);
                    __m128 _r0d = (__m128)__lsx_vreplvei_w(_r0nnn, 1);
                    __m128 _r0e = (__m128)__lsx_vreplvei_w(_r0nnn, 2);
                    __m128 _r0f = (__m128)__lsx_vreplvei_w(_r0nnn, 3);
                    __m128 _r0g = __lsx_vreplfr2vr_s(r0[16]);

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

                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);
                    __m128 _r13 = (__m128)__lsx_vreplvei_w(_r1, 3);
                    __m128 _r14 = (__m128)__lsx_vreplvei_w(_r1n, 0);
                    __m128 _r15 = (__m128)__lsx_vreplvei_w(_r1n, 1);
                    __m128 _r16 = (__m128)__lsx_vreplvei_w(_r1n, 2);
                    __m128 _r17 = (__m128)__lsx_vreplvei_w(_r1n, 3);
                    __m128 _r18 = (__m128)__lsx_vreplvei_w(_r1nn, 0);
                    __m128 _r19 = (__m128)__lsx_vreplvei_w(_r1nn, 1);
                    __m128 _r1a = (__m128)__lsx_vreplvei_w(_r1nn, 2);
                    __m128 _r1b = (__m128)__lsx_vreplvei_w(_r1nn, 3);
                    __m128 _r1c = (__m128)__lsx_vreplvei_w(_r1nnn, 0);
                    __m128 _r1d = (__m128)__lsx_vreplvei_w(_r1nnn, 1);
                    __m128 _r1e = (__m128)__lsx_vreplvei_w(_r1nnn, 2);
                    __m128 _r1f = (__m128)__lsx_vreplvei_w(_r1nnn, 3);
                    __m128 _r1g = __lsx_vreplfr2vr_s(r1[16]);

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

                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);
                    __m128 _r23 = (__m128)__lsx_vreplvei_w(_r2, 3);
                    __m128 _r24 = (__m128)__lsx_vreplvei_w(_r2n, 0);
                    __m128 _r25 = (__m128)__lsx_vreplvei_w(_r2n, 1);
                    __m128 _r26 = (__m128)__lsx_vreplvei_w(_r2n, 2);
                    __m128 _r27 = (__m128)__lsx_vreplvei_w(_r2n, 3);
                    __m128 _r28 = (__m128)__lsx_vreplvei_w(_r2nn, 0);
                    __m128 _r29 = (__m128)__lsx_vreplvei_w(_r2nn, 1);
                    __m128 _r2a = (__m128)__lsx_vreplvei_w(_r2nn, 2);
                    __m128 _r2b = (__m128)__lsx_vreplvei_w(_r2nn, 3);
                    __m128 _r2c = (__m128)__lsx_vreplvei_w(_r2nnn, 0);
                    __m128 _r2d = (__m128)__lsx_vreplvei_w(_r2nnn, 1);
                    __m128 _r2e = (__m128)__lsx_vreplvei_w(_r2nnn, 2);
                    __m128 _r2f = (__m128)__lsx_vreplvei_w(_r2nnn, 3);
                    __m128 _r2g = __lsx_vreplfr2vr_s(r2[16]);

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
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    __m128 _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);
                    __m128 _sum2 = (__m128)__lsx_vld(outptr0 + 4 * 2, 0);
                    __m128 _sum3 = (__m128)__lsx_vld(outptr0 + 4 * 3, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);

                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);
                    __m128 _r03 = (__m128)__lsx_vreplvei_w(_r0, 3);
                    __m128 _r04 = (__m128)__lsx_vreplvei_w(_r0n, 0);
                    __m128 _r05 = (__m128)__lsx_vreplvei_w(_r0n, 1);
                    __m128 _r06 = (__m128)__lsx_vreplvei_w(_r0n, 2);
                    __m128 _r07 = (__m128)__lsx_vreplvei_w(_r0n, 3);
                    __m128 _r08 = __lsx_vreplfr2vr_s(r0[8]);

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

                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);
                    __m128 _r13 = (__m128)__lsx_vreplvei_w(_r1, 3);
                    __m128 _r14 = (__m128)__lsx_vreplvei_w(_r1n, 0);
                    __m128 _r15 = (__m128)__lsx_vreplvei_w(_r1n, 1);
                    __m128 _r16 = (__m128)__lsx_vreplvei_w(_r1n, 2);
                    __m128 _r17 = (__m128)__lsx_vreplvei_w(_r1n, 3);
                    __m128 _r18 = __lsx_vreplfr2vr_s(r1[8]);

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

                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);
                    __m128 _r23 = (__m128)__lsx_vreplvei_w(_r2, 3);
                    __m128 _r24 = (__m128)__lsx_vreplvei_w(_r2n, 0);
                    __m128 _r25 = (__m128)__lsx_vreplvei_w(_r2n, 1);
                    __m128 _r26 = (__m128)__lsx_vreplvei_w(_r2n, 2);
                    __m128 _r27 = (__m128)__lsx_vreplvei_w(_r2n, 3);
                    __m128 _r28 = __lsx_vreplfr2vr_s(r2[8]);

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
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    __m128 _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);
                    __m128 _r03 = (__m128)__lsx_vreplvei_w(_r0, 3);
                    __m128 _r04 = __lsx_vreplfr2vr_s(r0[4]);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r02, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r03, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r04, _sum1);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);
                    __m128 _r13 = (__m128)__lsx_vreplvei_w(_r1, 3);
                    __m128 _r14 = __lsx_vreplfr2vr_s(r1[4]);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r12, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r13, _sum1);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r14, _sum1);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);
                    __m128 _r23 = (__m128)__lsx_vreplvei_w(_r2, 3);
                    __m128 _r24 = __lsx_vreplfr2vr_s(r2[4]);

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
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);

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

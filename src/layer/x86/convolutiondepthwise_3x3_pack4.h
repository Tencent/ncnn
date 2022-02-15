// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convdw3x3s1_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + g * 4) : _mm_set1_ps(0.f);

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        __m128 _k00 = _mm_load_ps(k0);
        __m128 _k01 = _mm_load_ps(k0 + 4);
        __m128 _k02 = _mm_load_ps(k0 + 8);
        __m128 _k10 = _mm_load_ps(k0 + 12);
        __m128 _k11 = _mm_load_ps(k0 + 16);
        __m128 _k12 = _mm_load_ps(k0 + 20);
        __m128 _k20 = _mm_load_ps(k0 + 24);
        __m128 _k21 = _mm_load_ps(k0 + 28);
        __m128 _k22 = _mm_load_ps(k0 + 32);

        int i = 0;

        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 7 < outw; j += 8)
            {
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 8);
                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 8);
                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 8);

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                __m128 _sum1 = _bias0;
                __m128 _r03 = _mm_load_ps(r0 + 12);
                __m128 _r13 = _mm_load_ps(r1 + 12);
                __m128 _r23 = _mm_load_ps(r2 + 12);
                _mm_store_ps(outptr0, _sum0);

                _sum1 = _mm_comp_fmadd_ps(_k00, _r01, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k01, _r02, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k02, _r03, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k10, _r11, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k11, _r12, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k12, _r13, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k20, _r21, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k21, _r22, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k22, _r23, _sum1);

                __m128 _sum2 = _bias0;
                __m128 _r04 = _mm_load_ps(r0 + 16);
                __m128 _r14 = _mm_load_ps(r1 + 16);
                __m128 _r24 = _mm_load_ps(r2 + 16);
                _mm_store_ps(outptr0 + 4, _sum1);

                _sum2 = _mm_comp_fmadd_ps(_k00, _r02, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k01, _r03, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k02, _r04, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k10, _r12, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k11, _r13, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k12, _r14, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k20, _r22, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k21, _r23, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k22, _r24, _sum2);

                __m128 _sum3 = _bias0;
                __m128 _r05 = _mm_load_ps(r0 + 20);
                __m128 _r15 = _mm_load_ps(r1 + 20);
                __m128 _r25 = _mm_load_ps(r2 + 20);
                _mm_store_ps(outptr0 + 8, _sum2);

                _sum3 = _mm_comp_fmadd_ps(_k00, _r03, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k01, _r04, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k02, _r05, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k10, _r13, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k11, _r14, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k12, _r15, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k20, _r23, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k21, _r24, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k22, _r25, _sum3);

                __m128 _sum4 = _bias0;
                __m128 _r06 = _mm_load_ps(r0 + 24);
                __m128 _r16 = _mm_load_ps(r1 + 24);
                __m128 _r26 = _mm_load_ps(r2 + 24);
                _mm_store_ps(outptr0 + 12, _sum3);

                _sum4 = _mm_comp_fmadd_ps(_k00, _r04, _sum4);
                _sum4 = _mm_comp_fmadd_ps(_k01, _r05, _sum4);
                _sum4 = _mm_comp_fmadd_ps(_k02, _r06, _sum4);
                _sum4 = _mm_comp_fmadd_ps(_k10, _r14, _sum4);
                _sum4 = _mm_comp_fmadd_ps(_k11, _r15, _sum4);
                _sum4 = _mm_comp_fmadd_ps(_k12, _r16, _sum4);
                _sum4 = _mm_comp_fmadd_ps(_k20, _r24, _sum4);
                _sum4 = _mm_comp_fmadd_ps(_k21, _r25, _sum4);
                _sum4 = _mm_comp_fmadd_ps(_k22, _r26, _sum4);

                __m128 _sum5 = _bias0;
                __m128 _r07 = _mm_load_ps(r0 + 28);
                __m128 _r17 = _mm_load_ps(r1 + 28);
                __m128 _r27 = _mm_load_ps(r2 + 28);
                _mm_store_ps(outptr0 + 16, _sum4);

                _sum5 = _mm_comp_fmadd_ps(_k00, _r05, _sum5);
                _sum5 = _mm_comp_fmadd_ps(_k01, _r06, _sum5);
                _sum5 = _mm_comp_fmadd_ps(_k02, _r07, _sum5);
                _sum5 = _mm_comp_fmadd_ps(_k10, _r15, _sum5);
                _sum5 = _mm_comp_fmadd_ps(_k11, _r16, _sum5);
                _sum5 = _mm_comp_fmadd_ps(_k12, _r17, _sum5);
                _sum5 = _mm_comp_fmadd_ps(_k20, _r25, _sum5);
                _sum5 = _mm_comp_fmadd_ps(_k21, _r26, _sum5);
                _sum5 = _mm_comp_fmadd_ps(_k22, _r27, _sum5);

                __m128 _sum6 = _bias0;
                __m128 _r08 = _mm_load_ps(r0 + 32);
                __m128 _r18 = _mm_load_ps(r1 + 32);
                __m128 _r28 = _mm_load_ps(r2 + 32);
                _mm_store_ps(outptr0 + 20, _sum5);

                _sum6 = _mm_comp_fmadd_ps(_k00, _r06, _sum6);
                _sum6 = _mm_comp_fmadd_ps(_k01, _r07, _sum6);
                _sum6 = _mm_comp_fmadd_ps(_k02, _r08, _sum6);
                _sum6 = _mm_comp_fmadd_ps(_k10, _r16, _sum6);
                _sum6 = _mm_comp_fmadd_ps(_k11, _r17, _sum6);
                _sum6 = _mm_comp_fmadd_ps(_k12, _r18, _sum6);
                _sum6 = _mm_comp_fmadd_ps(_k20, _r26, _sum6);
                _sum6 = _mm_comp_fmadd_ps(_k21, _r27, _sum6);
                _sum6 = _mm_comp_fmadd_ps(_k22, _r28, _sum6);

                __m128 _sum7 = _bias0;
                __m128 _r09 = _mm_load_ps(r0 + 36);
                __m128 _r19 = _mm_load_ps(r1 + 36);
                __m128 _r29 = _mm_load_ps(r2 + 36);
                _mm_store_ps(outptr0 + 24, _sum6);

                _sum7 = _mm_comp_fmadd_ps(_k00, _r07, _sum7);
                _sum7 = _mm_comp_fmadd_ps(_k01, _r08, _sum7);
                _sum7 = _mm_comp_fmadd_ps(_k02, _r09, _sum7);
                _sum7 = _mm_comp_fmadd_ps(_k10, _r17, _sum7);
                _sum7 = _mm_comp_fmadd_ps(_k11, _r18, _sum7);
                _sum7 = _mm_comp_fmadd_ps(_k12, _r19, _sum7);
                _sum7 = _mm_comp_fmadd_ps(_k20, _r27, _sum7);
                _sum7 = _mm_comp_fmadd_ps(_k21, _r28, _sum7);
                _sum7 = _mm_comp_fmadd_ps(_k22, _r29, _sum7);
                _mm_store_ps(outptr0 + 28, _sum7);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                outptr0 += 32;
            }
            for (; j + 3 < outw; j += 4)
            {
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 8);
                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 8);
                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 8);

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                __m128 _sum1 = _bias0;
                __m128 _r03 = _mm_load_ps(r0 + 12);
                __m128 _r13 = _mm_load_ps(r1 + 12);
                __m128 _r23 = _mm_load_ps(r2 + 12);
                _mm_store_ps(outptr0, _sum0);

                _sum1 = _mm_comp_fmadd_ps(_k00, _r01, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k01, _r02, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k02, _r03, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k10, _r11, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k11, _r12, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k12, _r13, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k20, _r21, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k21, _r22, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k22, _r23, _sum1);

                __m128 _sum2 = _bias0;
                __m128 _r04 = _mm_load_ps(r0 + 16);
                __m128 _r14 = _mm_load_ps(r1 + 16);
                __m128 _r24 = _mm_load_ps(r2 + 16);
                _mm_store_ps(outptr0 + 4, _sum1);

                _sum2 = _mm_comp_fmadd_ps(_k00, _r02, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k01, _r03, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k02, _r04, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k10, _r12, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k11, _r13, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k12, _r14, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k20, _r22, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k21, _r23, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k22, _r24, _sum2);

                __m128 _sum3 = _bias0;
                __m128 _r05 = _mm_load_ps(r0 + 20);
                __m128 _r15 = _mm_load_ps(r1 + 20);
                __m128 _r25 = _mm_load_ps(r2 + 20);
                _mm_store_ps(outptr0 + 8, _sum2);

                _sum3 = _mm_comp_fmadd_ps(_k00, _r03, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k01, _r04, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k02, _r05, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k10, _r13, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k11, _r14, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k12, _r15, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k20, _r23, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k21, _r24, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k22, _r25, _sum3);

                _mm_store_ps(outptr0 + 12, _sum3);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr0 += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 8);
                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 8);
                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 8);

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                __m128 _sum1 = _bias0;
                __m128 _r03 = _mm_load_ps(r0 + 12);
                __m128 _r13 = _mm_load_ps(r1 + 12);
                __m128 _r23 = _mm_load_ps(r2 + 12);
                _mm_store_ps(outptr0, _sum0);

                _sum1 = _mm_comp_fmadd_ps(_k00, _r01, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k01, _r02, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k02, _r03, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k10, _r11, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k11, _r12, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k12, _r13, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k20, _r21, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k21, _r22, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k22, _r23, _sum1);

                _mm_store_ps(outptr0 + 4, _sum1);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr0 += 8;
            }
            for (; j < outw; j++)
            {
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 8);
                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 8);
                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 8);

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                _mm_store_ps(outptr0, _sum0);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                outptr0 += 4;
            }

            r0 += 2 * 4;
            r1 += 2 * 4;
            r2 += 2 * 4;
        }
    }
}

static void convdw3x3s2_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * 4;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + g * 4) : _mm_set1_ps(0.f);

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        __m128 _k00 = _mm_load_ps(k0);
        __m128 _k01 = _mm_load_ps(k0 + 4);
        __m128 _k02 = _mm_load_ps(k0 + 8);
        __m128 _k10 = _mm_load_ps(k0 + 12);
        __m128 _k11 = _mm_load_ps(k0 + 16);
        __m128 _k12 = _mm_load_ps(k0 + 20);
        __m128 _k20 = _mm_load_ps(k0 + 24);
        __m128 _k21 = _mm_load_ps(k0 + 28);
        __m128 _k22 = _mm_load_ps(k0 + 32);

        int i = 0;

        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 8);
                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 8);
                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 8);

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                __m128 _sum1 = _bias0;
                __m128 _r03 = _mm_load_ps(r0 + 12);
                __m128 _r13 = _mm_load_ps(r1 + 12);
                __m128 _r23 = _mm_load_ps(r2 + 12);
                __m128 _r04 = _mm_load_ps(r0 + 16);
                __m128 _r14 = _mm_load_ps(r1 + 16);
                __m128 _r24 = _mm_load_ps(r2 + 16);
                _mm_store_ps(outptr0, _sum0);

                _sum1 = _mm_comp_fmadd_ps(_k00, _r02, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k01, _r03, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k02, _r04, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k10, _r12, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k11, _r13, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k12, _r14, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k20, _r22, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k21, _r23, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k22, _r24, _sum1);

                __m128 _sum2 = _bias0;
                __m128 _r05 = _mm_load_ps(r0 + 20);
                __m128 _r15 = _mm_load_ps(r1 + 20);
                __m128 _r25 = _mm_load_ps(r2 + 20);
                __m128 _r06 = _mm_load_ps(r0 + 24);
                __m128 _r16 = _mm_load_ps(r1 + 24);
                __m128 _r26 = _mm_load_ps(r2 + 24);
                _mm_store_ps(outptr0 + 4, _sum1);

                _sum2 = _mm_comp_fmadd_ps(_k00, _r04, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k01, _r05, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k02, _r06, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k10, _r14, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k11, _r15, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k12, _r16, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k20, _r24, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k21, _r25, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_k22, _r26, _sum2);

                __m128 _sum3 = _bias0;
                __m128 _r07 = _mm_load_ps(r0 + 28);
                __m128 _r17 = _mm_load_ps(r1 + 28);
                __m128 _r27 = _mm_load_ps(r2 + 28);
                __m128 _r08 = _mm_load_ps(r0 + 32);
                __m128 _r18 = _mm_load_ps(r1 + 32);
                __m128 _r28 = _mm_load_ps(r2 + 32);
                _mm_store_ps(outptr0 + 8, _sum2);

                _sum3 = _mm_comp_fmadd_ps(_k00, _r06, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k01, _r07, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k02, _r08, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k10, _r16, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k11, _r17, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k12, _r18, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k20, _r26, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k21, _r27, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_k22, _r28, _sum3);

                _mm_store_ps(outptr0 + 12, _sum3);

                r0 += 2 * 16;
                r1 += 2 * 16;
                r2 += 2 * 16;
                outptr0 += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 8);
                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 8);
                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 8);

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                __m128 _sum1 = _bias0;
                __m128 _r03 = _mm_load_ps(r0 + 12);
                __m128 _r13 = _mm_load_ps(r1 + 12);
                __m128 _r23 = _mm_load_ps(r2 + 12);
                __m128 _r04 = _mm_load_ps(r0 + 16);
                __m128 _r14 = _mm_load_ps(r1 + 16);
                __m128 _r24 = _mm_load_ps(r2 + 16);
                _mm_store_ps(outptr0, _sum0);

                _sum1 = _mm_comp_fmadd_ps(_k00, _r02, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k01, _r03, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k02, _r04, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k10, _r12, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k11, _r13, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k12, _r14, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k20, _r22, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k21, _r23, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k22, _r24, _sum1);

                _mm_store_ps(outptr0 + 4, _sum1);

                r0 += 2 * 8;
                r1 += 2 * 8;
                r2 += 2 * 8;
                outptr0 += 8;
            }
            for (; j < outw; j++)
            {
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 8);
                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 8);
                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 8);

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);
                _mm_store_ps(outptr0, _sum0);
                r0 += 2 * 4;
                r1 += 2 * 4;
                r2 += 2 * 4;
                outptr0 += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

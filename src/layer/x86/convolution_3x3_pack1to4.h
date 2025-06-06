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

static void conv3x3s1_pack1to4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;
    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);

        __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + p * 4) : _mm_set1_ps(0.f);
        __m128 _bias1 = bias ? _mm_loadu_ps((const float*)bias + (p + 1) * 4) : _mm_set1_ps(0.f);
        out0.fill(_bias0);
        out1.fill(_bias1);

        const float* k0 = kernel.channel(p);
        const float* k1 = kernel.channel(p + 1);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m128 _k00_0 = _mm_loadu_ps(k0);
            __m128 _k01_0 = _mm_loadu_ps(k0 + 4);
            __m128 _k02_0 = _mm_loadu_ps(k0 + 8);
            __m128 _k10_0 = _mm_loadu_ps(k0 + 12);
            __m128 _k11_0 = _mm_loadu_ps(k0 + 16);
            __m128 _k12_0 = _mm_loadu_ps(k0 + 20);
            __m128 _k20_0 = _mm_loadu_ps(k0 + 24);
            __m128 _k21_0 = _mm_loadu_ps(k0 + 28);
            __m128 _k22_0 = _mm_loadu_ps(k0 + 32);

            __m128 _k00_1 = _mm_loadu_ps(k1);
            __m128 _k01_1 = _mm_loadu_ps(k1 + 4);
            __m128 _k02_1 = _mm_loadu_ps(k1 + 8);
            __m128 _k10_1 = _mm_loadu_ps(k1 + 12);
            __m128 _k11_1 = _mm_loadu_ps(k1 + 16);
            __m128 _k12_1 = _mm_loadu_ps(k1 + 20);
            __m128 _k20_1 = _mm_loadu_ps(k1 + 24);
            __m128 _k21_1 = _mm_loadu_ps(k1 + 28);
            __m128 _k22_1 = _mm_loadu_ps(k1 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);
                    __m128 _sum10 = _mm_loadu_ps(outptr1);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _sum10 = _mm_comp_fmadd_ps(_r01, _k00_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r02, _k01_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r03, _k02_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r11, _k10_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r12, _k11_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r13, _k12_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r21, _k20_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r22, _k21_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r23, _k22_1, _sum10);

                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr1, _sum10);

                    __m128 _sum01 = _mm_loadu_ps(outptr0 + 4);
                    __m128 _sum11 = _mm_loadu_ps(outptr1 + 4);

                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));

                    _sum01 = _mm_comp_fmadd_ps(_r02, _k00_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r03, _k01_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r04, _k02_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r12, _k10_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r13, _k11_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r14, _k12_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r22, _k20_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r23, _k21_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r24, _k22_0, _sum01);

                    _sum11 = _mm_comp_fmadd_ps(_r02, _k00_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r03, _k01_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r04, _k02_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r12, _k10_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r13, _k11_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r14, _k12_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r22, _k20_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r23, _k21_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r24, _k22_1, _sum11);

                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr1 + 4, _sum11);

                    __m128 _sum02 = _mm_loadu_ps(outptr0 + 8);
                    __m128 _sum12 = _mm_loadu_ps(outptr1 + 8);

                    __m128 _r05 = _mm_set1_ps(*(r0 + 4));
                    __m128 _r15 = _mm_set1_ps(*(r1 + 4));
                    __m128 _r25 = _mm_set1_ps(*(r2 + 4));

                    _sum02 = _mm_comp_fmadd_ps(_r03, _k00_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r04, _k01_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r05, _k02_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r13, _k10_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r14, _k11_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r15, _k12_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r23, _k20_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r24, _k21_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r25, _k22_0, _sum02);

                    _sum12 = _mm_comp_fmadd_ps(_r03, _k00_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r04, _k01_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r05, _k02_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r13, _k10_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r14, _k11_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r15, _k12_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r23, _k20_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r24, _k21_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r25, _k22_1, _sum12);

                    _mm_storeu_ps(outptr0 + 8, _sum02);
                    _mm_storeu_ps(outptr1 + 8, _sum12);

                    __m128 _r06 = _mm_set1_ps(*(r0 + 5));
                    __m128 _r16 = _mm_set1_ps(*(r1 + 5));
                    __m128 _r26 = _mm_set1_ps(*(r2 + 5));

                    __m128 _sum03 = _mm_loadu_ps(outptr0 + 12);
                    __m128 _sum13 = _mm_loadu_ps(outptr1 + 12);

                    _sum03 = _mm_comp_fmadd_ps(_r04, _k00_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r05, _k01_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r06, _k02_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r14, _k10_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r15, _k11_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r16, _k12_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r24, _k20_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r25, _k21_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r26, _k22_0, _sum03);

                    _sum13 = _mm_comp_fmadd_ps(_r04, _k00_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r05, _k01_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r06, _k02_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r14, _k10_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r15, _k11_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r16, _k12_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r24, _k20_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r25, _k21_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r26, _k22_1, _sum13);

                    _mm_storeu_ps(outptr0 + 12, _sum03);
                    _mm_storeu_ps(outptr1 + 12, _sum13);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr0 += 16;
                    outptr1 += 16;
                }

                for (; j + 1 < outw; j += 2)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);
                    __m128 _sum10 = _mm_loadu_ps(outptr1);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _sum10 = _mm_comp_fmadd_ps(_r01, _k00_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r02, _k01_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r03, _k02_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r11, _k10_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r12, _k11_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r13, _k12_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r21, _k20_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r22, _k21_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r23, _k22_1, _sum10);

                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr1, _sum10);

                    __m128 _sum01 = _mm_loadu_ps(outptr0 + 4);
                    __m128 _sum11 = _mm_loadu_ps(outptr1 + 4);

                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));

                    _sum01 = _mm_comp_fmadd_ps(_r02, _k00_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r03, _k01_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r04, _k02_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r12, _k10_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r13, _k11_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r14, _k12_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r22, _k20_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r23, _k21_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r24, _k22_0, _sum01);

                    _sum11 = _mm_comp_fmadd_ps(_r02, _k00_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r03, _k01_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r04, _k02_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r12, _k10_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r13, _k11_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r14, _k12_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r22, _k20_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r23, _k21_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r24, _k22_1, _sum11);

                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr1 + 4, _sum11);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 8;
                    outptr1 += 8;
                }

                for (; j < outw; j++)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);
                    __m128 _sum10 = _mm_loadu_ps(outptr1);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _sum10 = _mm_comp_fmadd_ps(_r01, _k00_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r02, _k01_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r03, _k02_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r11, _k10_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r12, _k11_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r13, _k12_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r21, _k20_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r22, _k21_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r23, _k22_1, _sum10);

                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr1, _sum10);

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                    outptr0 += 4;
                    outptr1 += 4;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * 4;
            k1 += 9 * 4;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + p * 4) : _mm_set1_ps(0.f);
        out0.fill(_bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m128 _k00 = _mm_loadu_ps(k0);
            __m128 _k01 = _mm_loadu_ps(k0 + 4);
            __m128 _k02 = _mm_loadu_ps(k0 + 8);
            __m128 _k10 = _mm_loadu_ps(k0 + 12);
            __m128 _k11 = _mm_loadu_ps(k0 + 16);
            __m128 _k12 = _mm_loadu_ps(k0 + 20);
            __m128 _k20 = _mm_loadu_ps(k0 + 24);
            __m128 _k21 = _mm_loadu_ps(k0 + 28);
            __m128 _k22 = _mm_loadu_ps(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    __m128 _sum0 = _mm_loadu_ps(outptr0);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum0 = _mm_comp_fmadd_ps(_r01, _k00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r02, _k01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r03, _k02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r11, _k10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r12, _k11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r13, _k12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r21, _k20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r22, _k21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r23, _k22, _sum0);

                    __m128 _sum1 = _mm_loadu_ps(outptr0 + 4);
                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));
                    _mm_storeu_ps(outptr0, _sum0);

                    _sum1 = _mm_comp_fmadd_ps(_r02, _k00, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r03, _k01, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r04, _k02, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r12, _k10, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r13, _k11, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r14, _k12, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r22, _k20, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r23, _k21, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r24, _k22, _sum1);

                    __m128 _sum2 = _mm_loadu_ps(outptr0 + 8);
                    __m128 _r05 = _mm_set1_ps(*(r0 + 4));
                    __m128 _r15 = _mm_set1_ps(*(r1 + 4));
                    __m128 _r25 = _mm_set1_ps(*(r2 + 4));
                    _mm_storeu_ps(outptr0 + 4, _sum1);

                    _sum2 = _mm_comp_fmadd_ps(_r03, _k00, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_r04, _k01, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_r05, _k02, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_r13, _k10, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_r14, _k11, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_r15, _k12, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_r23, _k20, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_r24, _k21, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_r25, _k22, _sum2);

                    __m128 _sum3 = _mm_loadu_ps(outptr0 + 12);
                    __m128 _r06 = _mm_set1_ps(*(r0 + 5));
                    __m128 _r16 = _mm_set1_ps(*(r1 + 5));
                    __m128 _r26 = _mm_set1_ps(*(r2 + 5));
                    _mm_storeu_ps(outptr0 + 8, _sum2);

                    _sum3 = _mm_comp_fmadd_ps(_r04, _k00, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_r05, _k01, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_r06, _k02, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_r14, _k10, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_r15, _k11, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_r16, _k12, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_r24, _k20, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_r25, _k21, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_r26, _k22, _sum3);

                    _mm_storeu_ps(outptr0 + 12, _sum3);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr0 += 16;
                }
                for (; j + 1 < outw; j += 2)
                {
                    __m128 _sum0 = _mm_loadu_ps(outptr0);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum0 = _mm_comp_fmadd_ps(_r01, _k00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r02, _k01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r03, _k02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r11, _k10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r12, _k11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r13, _k12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r21, _k20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r22, _k21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r23, _k22, _sum0);

                    __m128 _sum1 = _mm_loadu_ps(outptr0 + 4);
                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));
                    _mm_storeu_ps(outptr0, _sum0);

                    _sum1 = _mm_comp_fmadd_ps(_r02, _k00, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r03, _k01, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r04, _k02, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r12, _k10, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r13, _k11, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r14, _k12, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r22, _k20, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r23, _k21, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_r24, _k22, _sum1);

                    _mm_storeu_ps(outptr0 + 4, _sum1);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 8;
                }
                for (; j < outw; j++)
                {
                    __m128 _sum0 = _mm_loadu_ps(outptr0);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum0 = _mm_comp_fmadd_ps(_r01, _k00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r02, _k01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r03, _k02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r11, _k10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r12, _k11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r13, _k12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r21, _k20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r22, _k21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_r23, _k22, _sum0);

                    _mm_storeu_ps(outptr0, _sum0);
                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                    outptr0 += 4;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * 4;
        }
    }
}

static void conv3x3s2_pack1to4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* bias = _bias;
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);

        __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + p * 4) : _mm_set1_ps(0.f);
        __m128 _bias1 = bias ? _mm_loadu_ps((const float*)bias + (p + 1) * 4) : _mm_set1_ps(0.f);
        out0.fill(_bias0);
        out1.fill(_bias1);

        const float* k0 = kernel.channel(p);
        const float* k1 = kernel.channel(p + 1);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m128 _k00_0 = _mm_loadu_ps(k0);
            __m128 _k01_0 = _mm_loadu_ps(k0 + 4);
            __m128 _k02_0 = _mm_loadu_ps(k0 + 8);
            __m128 _k10_0 = _mm_loadu_ps(k0 + 12);
            __m128 _k11_0 = _mm_loadu_ps(k0 + 16);
            __m128 _k12_0 = _mm_loadu_ps(k0 + 20);
            __m128 _k20_0 = _mm_loadu_ps(k0 + 24);
            __m128 _k21_0 = _mm_loadu_ps(k0 + 28);
            __m128 _k22_0 = _mm_loadu_ps(k0 + 32);

            __m128 _k00_1 = _mm_loadu_ps(k1);
            __m128 _k01_1 = _mm_loadu_ps(k1 + 4);
            __m128 _k02_1 = _mm_loadu_ps(k1 + 8);
            __m128 _k10_1 = _mm_loadu_ps(k1 + 12);
            __m128 _k11_1 = _mm_loadu_ps(k1 + 16);
            __m128 _k12_1 = _mm_loadu_ps(k1 + 20);
            __m128 _k20_1 = _mm_loadu_ps(k1 + 24);
            __m128 _k21_1 = _mm_loadu_ps(k1 + 28);
            __m128 _k22_1 = _mm_loadu_ps(k1 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);
                    __m128 _sum10 = _mm_loadu_ps(outptr1);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _sum10 = _mm_comp_fmadd_ps(_r01, _k00_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r02, _k01_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r03, _k02_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r11, _k10_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r12, _k11_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r13, _k12_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r21, _k20_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r22, _k21_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r23, _k22_1, _sum10);

                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr1, _sum10);

                    __m128 _sum01 = _mm_loadu_ps(outptr0 + 4);
                    __m128 _sum11 = _mm_loadu_ps(outptr1 + 4);

                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));
                    __m128 _r05 = _mm_set1_ps(*(r0 + 4));
                    __m128 _r15 = _mm_set1_ps(*(r1 + 4));
                    __m128 _r25 = _mm_set1_ps(*(r2 + 4));

                    _sum01 = _mm_comp_fmadd_ps(_r03, _k00_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r04, _k01_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r05, _k02_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r13, _k10_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r14, _k11_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r15, _k12_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r23, _k20_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r24, _k21_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r25, _k22_0, _sum01);

                    _sum11 = _mm_comp_fmadd_ps(_r03, _k00_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r04, _k01_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r05, _k02_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r13, _k10_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r14, _k11_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r15, _k12_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r23, _k20_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r24, _k21_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r25, _k22_1, _sum11);

                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr1 + 4, _sum11);

                    __m128 _sum02 = _mm_loadu_ps(outptr0 + 8);
                    __m128 _sum12 = _mm_loadu_ps(outptr1 + 8);

                    __m128 _r06 = _mm_set1_ps(*(r0 + 5));
                    __m128 _r16 = _mm_set1_ps(*(r1 + 5));
                    __m128 _r26 = _mm_set1_ps(*(r2 + 5));
                    __m128 _r07 = _mm_set1_ps(*(r0 + 6));
                    __m128 _r17 = _mm_set1_ps(*(r1 + 6));
                    __m128 _r27 = _mm_set1_ps(*(r2 + 6));

                    _sum02 = _mm_comp_fmadd_ps(_r05, _k00_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r06, _k01_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r07, _k02_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r15, _k10_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r16, _k11_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r17, _k12_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r25, _k20_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r26, _k21_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r27, _k22_0, _sum02);

                    _sum12 = _mm_comp_fmadd_ps(_r05, _k00_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r06, _k01_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r07, _k02_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r15, _k10_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r16, _k11_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r17, _k12_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r25, _k20_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r26, _k21_1, _sum12);
                    _sum12 = _mm_comp_fmadd_ps(_r27, _k22_1, _sum12);

                    _mm_storeu_ps(outptr0 + 8, _sum02);
                    _mm_storeu_ps(outptr1 + 8, _sum12);

                    __m128 _r08 = _mm_set1_ps(*(r0 + 7));
                    __m128 _r18 = _mm_set1_ps(*(r1 + 7));
                    __m128 _r28 = _mm_set1_ps(*(r2 + 7));
                    __m128 _r09 = _mm_set1_ps(*(r0 + 8));
                    __m128 _r19 = _mm_set1_ps(*(r1 + 8));
                    __m128 _r29 = _mm_set1_ps(*(r2 + 8));

                    __m128 _sum03 = _mm_loadu_ps(outptr0 + 12);
                    __m128 _sum13 = _mm_loadu_ps(outptr1 + 12);

                    _sum03 = _mm_comp_fmadd_ps(_r07, _k00_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r08, _k01_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r09, _k02_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r17, _k10_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r18, _k11_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r19, _k12_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r27, _k20_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r28, _k21_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r29, _k22_0, _sum03);

                    _sum13 = _mm_comp_fmadd_ps(_r07, _k00_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r08, _k01_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r09, _k02_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r17, _k10_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r18, _k11_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r19, _k12_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r27, _k20_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r28, _k21_1, _sum13);
                    _sum13 = _mm_comp_fmadd_ps(_r29, _k22_1, _sum13);

                    _mm_storeu_ps(outptr0 + 12, _sum03);
                    _mm_storeu_ps(outptr1 + 12, _sum13);
                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr0 += 16;
                    outptr1 += 16;
                }

                for (; j + 1 < outw; j += 2)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);
                    __m128 _sum10 = _mm_loadu_ps(outptr1);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _sum10 = _mm_comp_fmadd_ps(_r01, _k00_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r02, _k01_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r03, _k02_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r11, _k10_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r12, _k11_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r13, _k12_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r21, _k20_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r22, _k21_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r23, _k22_1, _sum10);

                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr1, _sum10);

                    __m128 _sum01 = _mm_loadu_ps(outptr0 + 4);
                    __m128 _sum11 = _mm_loadu_ps(outptr1 + 4);

                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));
                    __m128 _r05 = _mm_set1_ps(*(r0 + 4));
                    __m128 _r15 = _mm_set1_ps(*(r1 + 4));
                    __m128 _r25 = _mm_set1_ps(*(r2 + 4));

                    _sum01 = _mm_comp_fmadd_ps(_r03, _k00_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r04, _k01_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r05, _k02_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r13, _k10_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r14, _k11_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r15, _k12_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r23, _k20_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r24, _k21_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r25, _k22_0, _sum01);

                    _sum11 = _mm_comp_fmadd_ps(_r03, _k00_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r04, _k01_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r05, _k02_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r13, _k10_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r14, _k11_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r15, _k12_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r23, _k20_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r24, _k21_1, _sum11);
                    _sum11 = _mm_comp_fmadd_ps(_r25, _k22_1, _sum11);

                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr1 + 4, _sum11);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr0 += 8;
                    outptr1 += 8;
                }
                for (; j < outw; j++)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);
                    __m128 _sum10 = _mm_loadu_ps(outptr1);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _sum10 = _mm_comp_fmadd_ps(_r01, _k00_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r02, _k01_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r03, _k02_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r11, _k10_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r12, _k11_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r13, _k12_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r21, _k20_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r22, _k21_1, _sum10);
                    _sum10 = _mm_comp_fmadd_ps(_r23, _k22_1, _sum10);

                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr1, _sum10);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 4;
                    outptr1 += 4;
                }
                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * 4;
            k1 += 9 * 4;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + p * 4) : _mm_set1_ps(0.f);
        out0.fill(_bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m128 _k00_0 = _mm_loadu_ps(k0);
            __m128 _k01_0 = _mm_loadu_ps(k0 + 4);
            __m128 _k02_0 = _mm_loadu_ps(k0 + 8);
            __m128 _k10_0 = _mm_loadu_ps(k0 + 12);
            __m128 _k11_0 = _mm_loadu_ps(k0 + 16);
            __m128 _k12_0 = _mm_loadu_ps(k0 + 20);
            __m128 _k20_0 = _mm_loadu_ps(k0 + 24);
            __m128 _k21_0 = _mm_loadu_ps(k0 + 28);
            __m128 _k22_0 = _mm_loadu_ps(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _mm_storeu_ps(outptr0, _sum00);

                    __m128 _sum01 = _mm_loadu_ps(outptr0 + 4);

                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));
                    __m128 _r05 = _mm_set1_ps(*(r0 + 4));
                    __m128 _r15 = _mm_set1_ps(*(r1 + 4));
                    __m128 _r25 = _mm_set1_ps(*(r2 + 4));

                    _sum01 = _mm_comp_fmadd_ps(_r03, _k00_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r04, _k01_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r05, _k02_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r13, _k10_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r14, _k11_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r15, _k12_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r23, _k20_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r24, _k21_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r25, _k22_0, _sum01);

                    _mm_storeu_ps(outptr0 + 4, _sum01);

                    __m128 _sum02 = _mm_loadu_ps(outptr0 + 8);

                    __m128 _r06 = _mm_set1_ps(*(r0 + 5));
                    __m128 _r16 = _mm_set1_ps(*(r1 + 5));
                    __m128 _r26 = _mm_set1_ps(*(r2 + 5));
                    __m128 _r07 = _mm_set1_ps(*(r0 + 6));
                    __m128 _r17 = _mm_set1_ps(*(r1 + 6));
                    __m128 _r27 = _mm_set1_ps(*(r2 + 6));

                    _sum02 = _mm_comp_fmadd_ps(_r05, _k00_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r06, _k01_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r07, _k02_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r15, _k10_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r16, _k11_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r17, _k12_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r25, _k20_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r26, _k21_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r27, _k22_0, _sum02);

                    _mm_storeu_ps(outptr0 + 8, _sum02);

                    __m128 _r08 = _mm_set1_ps(*(r0 + 7));
                    __m128 _r18 = _mm_set1_ps(*(r1 + 7));
                    __m128 _r28 = _mm_set1_ps(*(r2 + 7));
                    __m128 _r09 = _mm_set1_ps(*(r0 + 8));
                    __m128 _r19 = _mm_set1_ps(*(r1 + 8));
                    __m128 _r29 = _mm_set1_ps(*(r2 + 8));

                    __m128 _sum03 = _mm_loadu_ps(outptr0 + 12);

                    _sum03 = _mm_comp_fmadd_ps(_r07, _k00_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r08, _k01_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r09, _k02_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r17, _k10_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r18, _k11_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r19, _k12_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r27, _k20_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r28, _k21_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r29, _k22_0, _sum03);

                    _mm_storeu_ps(outptr0 + 12, _sum03);

                    __m128 _r010 = _mm_set1_ps(*(r0 + 9));
                    __m128 _r110 = _mm_set1_ps(*(r1 + 9));
                    __m128 _r210 = _mm_set1_ps(*(r2 + 9));
                    __m128 _r011 = _mm_set1_ps(*(r0 + 10));
                    __m128 _r111 = _mm_set1_ps(*(r1 + 10));
                    __m128 _r211 = _mm_set1_ps(*(r2 + 10));

                    __m128 _sum04 = _mm_loadu_ps(outptr0 + 16);

                    _sum04 = _mm_comp_fmadd_ps(_r09, _k00_0, _sum04);
                    _sum04 = _mm_comp_fmadd_ps(_r010, _k01_0, _sum04);
                    _sum04 = _mm_comp_fmadd_ps(_r011, _k02_0, _sum04);
                    _sum04 = _mm_comp_fmadd_ps(_r19, _k10_0, _sum04);
                    _sum04 = _mm_comp_fmadd_ps(_r110, _k11_0, _sum04);
                    _sum04 = _mm_comp_fmadd_ps(_r111, _k12_0, _sum04);
                    _sum04 = _mm_comp_fmadd_ps(_r29, _k20_0, _sum04);
                    _sum04 = _mm_comp_fmadd_ps(_r210, _k21_0, _sum04);
                    _sum04 = _mm_comp_fmadd_ps(_r211, _k22_0, _sum04);

                    _mm_storeu_ps(outptr0 + 16, _sum04);

                    __m128 _r012 = _mm_set1_ps(*(r0 + 11));
                    __m128 _r112 = _mm_set1_ps(*(r1 + 11));
                    __m128 _r212 = _mm_set1_ps(*(r2 + 11));
                    __m128 _r013 = _mm_set1_ps(*(r0 + 12));
                    __m128 _r113 = _mm_set1_ps(*(r1 + 12));
                    __m128 _r213 = _mm_set1_ps(*(r2 + 12));

                    __m128 _sum05 = _mm_loadu_ps(outptr0 + 20);

                    _sum05 = _mm_comp_fmadd_ps(_r011, _k00_0, _sum05);
                    _sum05 = _mm_comp_fmadd_ps(_r012, _k01_0, _sum05);
                    _sum05 = _mm_comp_fmadd_ps(_r013, _k02_0, _sum05);
                    _sum05 = _mm_comp_fmadd_ps(_r111, _k10_0, _sum05);
                    _sum05 = _mm_comp_fmadd_ps(_r112, _k11_0, _sum05);
                    _sum05 = _mm_comp_fmadd_ps(_r113, _k12_0, _sum05);
                    _sum05 = _mm_comp_fmadd_ps(_r211, _k20_0, _sum05);
                    _sum05 = _mm_comp_fmadd_ps(_r212, _k21_0, _sum05);
                    _sum05 = _mm_comp_fmadd_ps(_r213, _k22_0, _sum05);

                    _mm_storeu_ps(outptr0 + 20, _sum05);

                    __m128 _r014 = _mm_set1_ps(*(r0 + 13));
                    __m128 _r114 = _mm_set1_ps(*(r1 + 13));
                    __m128 _r214 = _mm_set1_ps(*(r2 + 13));
                    __m128 _r015 = _mm_set1_ps(*(r0 + 14));
                    __m128 _r115 = _mm_set1_ps(*(r1 + 14));
                    __m128 _r215 = _mm_set1_ps(*(r2 + 14));

                    __m128 _sum06 = _mm_loadu_ps(outptr0 + 24);

                    _sum06 = _mm_comp_fmadd_ps(_r013, _k00_0, _sum06);
                    _sum06 = _mm_comp_fmadd_ps(_r014, _k01_0, _sum06);
                    _sum06 = _mm_comp_fmadd_ps(_r015, _k02_0, _sum06);
                    _sum06 = _mm_comp_fmadd_ps(_r113, _k10_0, _sum06);
                    _sum06 = _mm_comp_fmadd_ps(_r114, _k11_0, _sum06);
                    _sum06 = _mm_comp_fmadd_ps(_r115, _k12_0, _sum06);
                    _sum06 = _mm_comp_fmadd_ps(_r213, _k20_0, _sum06);
                    _sum06 = _mm_comp_fmadd_ps(_r214, _k21_0, _sum06);
                    _sum06 = _mm_comp_fmadd_ps(_r215, _k22_0, _sum06);

                    _mm_storeu_ps(outptr0 + 24, _sum06);

                    __m128 _r016 = _mm_set1_ps(*(r0 + 15));
                    __m128 _r116 = _mm_set1_ps(*(r1 + 15));
                    __m128 _r216 = _mm_set1_ps(*(r2 + 15));
                    __m128 _r017 = _mm_set1_ps(*(r0 + 16));
                    __m128 _r117 = _mm_set1_ps(*(r1 + 16));
                    __m128 _r217 = _mm_set1_ps(*(r2 + 16));

                    __m128 _sum07 = _mm_loadu_ps(outptr0 + 28);

                    _sum07 = _mm_comp_fmadd_ps(_r015, _k00_0, _sum07);
                    _sum07 = _mm_comp_fmadd_ps(_r016, _k01_0, _sum07);
                    _sum07 = _mm_comp_fmadd_ps(_r017, _k02_0, _sum07);
                    _sum07 = _mm_comp_fmadd_ps(_r115, _k10_0, _sum07);
                    _sum07 = _mm_comp_fmadd_ps(_r116, _k11_0, _sum07);
                    _sum07 = _mm_comp_fmadd_ps(_r117, _k12_0, _sum07);
                    _sum07 = _mm_comp_fmadd_ps(_r215, _k20_0, _sum07);
                    _sum07 = _mm_comp_fmadd_ps(_r216, _k21_0, _sum07);
                    _sum07 = _mm_comp_fmadd_ps(_r217, _k22_0, _sum07);

                    _mm_storeu_ps(outptr0 + 28, _sum07);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    outptr0 += 32;
                }

                for (; j + 3 < outw; j += 4)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _mm_storeu_ps(outptr0, _sum00);

                    __m128 _sum01 = _mm_loadu_ps(outptr0 + 4);

                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));
                    __m128 _r05 = _mm_set1_ps(*(r0 + 4));
                    __m128 _r15 = _mm_set1_ps(*(r1 + 4));
                    __m128 _r25 = _mm_set1_ps(*(r2 + 4));

                    _sum01 = _mm_comp_fmadd_ps(_r03, _k00_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r04, _k01_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r05, _k02_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r13, _k10_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r14, _k11_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r15, _k12_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r23, _k20_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r24, _k21_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r25, _k22_0, _sum01);

                    _mm_storeu_ps(outptr0 + 4, _sum01);

                    __m128 _sum02 = _mm_loadu_ps(outptr0 + 8);

                    __m128 _r06 = _mm_set1_ps(*(r0 + 5));
                    __m128 _r16 = _mm_set1_ps(*(r1 + 5));
                    __m128 _r26 = _mm_set1_ps(*(r2 + 5));
                    __m128 _r07 = _mm_set1_ps(*(r0 + 6));
                    __m128 _r17 = _mm_set1_ps(*(r1 + 6));
                    __m128 _r27 = _mm_set1_ps(*(r2 + 6));

                    _sum02 = _mm_comp_fmadd_ps(_r05, _k00_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r06, _k01_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r07, _k02_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r15, _k10_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r16, _k11_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r17, _k12_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r25, _k20_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r26, _k21_0, _sum02);
                    _sum02 = _mm_comp_fmadd_ps(_r27, _k22_0, _sum02);

                    _mm_storeu_ps(outptr0 + 8, _sum02);

                    __m128 _r08 = _mm_set1_ps(*(r0 + 7));
                    __m128 _r18 = _mm_set1_ps(*(r1 + 7));
                    __m128 _r28 = _mm_set1_ps(*(r2 + 7));
                    __m128 _r09 = _mm_set1_ps(*(r0 + 8));
                    __m128 _r19 = _mm_set1_ps(*(r1 + 8));
                    __m128 _r29 = _mm_set1_ps(*(r2 + 8));

                    __m128 _sum03 = _mm_loadu_ps(outptr0 + 12);

                    _sum03 = _mm_comp_fmadd_ps(_r07, _k00_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r08, _k01_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r09, _k02_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r17, _k10_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r18, _k11_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r19, _k12_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r27, _k20_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r28, _k21_0, _sum03);
                    _sum03 = _mm_comp_fmadd_ps(_r29, _k22_0, _sum03);

                    _mm_storeu_ps(outptr0 + 12, _sum03);
                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr0 += 16;
                }

                for (; j + 1 < outw; j += 2)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);

                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);

                    _mm_storeu_ps(outptr0, _sum00);

                    __m128 _sum01 = _mm_loadu_ps(outptr0 + 4);

                    __m128 _r04 = _mm_set1_ps(*(r0 + 3));
                    __m128 _r14 = _mm_set1_ps(*(r1 + 3));
                    __m128 _r24 = _mm_set1_ps(*(r2 + 3));
                    __m128 _r05 = _mm_set1_ps(*(r0 + 4));
                    __m128 _r15 = _mm_set1_ps(*(r1 + 4));
                    __m128 _r25 = _mm_set1_ps(*(r2 + 4));

                    _sum01 = _mm_comp_fmadd_ps(_r03, _k00_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r04, _k01_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r05, _k02_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r13, _k10_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r14, _k11_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r15, _k12_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r23, _k20_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r24, _k21_0, _sum01);
                    _sum01 = _mm_comp_fmadd_ps(_r25, _k22_0, _sum01);

                    _mm_storeu_ps(outptr0 + 4, _sum01);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr0 += 8;
                }
                for (; j < outw; j++)
                {
                    __m128 _sum00 = _mm_loadu_ps(outptr0);
                    __m128 _r01 = _mm_set1_ps(*(r0));
                    __m128 _r02 = _mm_set1_ps(*(r0 + 1));
                    __m128 _r03 = _mm_set1_ps(*(r0 + 2));
                    __m128 _r11 = _mm_set1_ps(*(r1));
                    __m128 _r12 = _mm_set1_ps(*(r1 + 1));
                    __m128 _r13 = _mm_set1_ps(*(r1 + 2));
                    __m128 _r21 = _mm_set1_ps(*(r2));
                    __m128 _r22 = _mm_set1_ps(*(r2 + 1));
                    __m128 _r23 = _mm_set1_ps(*(r2 + 2));

                    _sum00 = _mm_comp_fmadd_ps(_r01, _k00_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r02, _k01_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r03, _k02_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r11, _k10_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r12, _k11_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r13, _k12_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r21, _k20_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r22, _k21_0, _sum00);
                    _sum00 = _mm_comp_fmadd_ps(_r23, _k22_0, _sum00);
                    _mm_storeu_ps(outptr0, _sum00);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 4;
                }
                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * 4;
        }
    }
}

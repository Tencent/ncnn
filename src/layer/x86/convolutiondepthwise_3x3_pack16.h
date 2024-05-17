// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convdw3x3s1_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __m512 _bias0 = bias ? _mm512_loadu_ps((const float*)bias + g * 16) : _mm512_setzero_ps();

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);
        float* outptr1 = out.row(1);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        const float* r3 = img0.row(3);

        __m512 _k00 = _mm512_load_ps(k0);
        __m512 _k01 = _mm512_load_ps(k0 + 16);
        __m512 _k02 = _mm512_load_ps(k0 + 32);
        __m512 _k10 = _mm512_load_ps(k0 + 48);
        __m512 _k11 = _mm512_load_ps(k0 + 64);
        __m512 _k12 = _mm512_load_ps(k0 + 80);
        __m512 _k20 = _mm512_load_ps(k0 + 96);
        __m512 _k21 = _mm512_load_ps(k0 + 112);
        __m512 _k22 = _mm512_load_ps(k0 + 128);

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m512 _sum00 = _bias0;
                __m512 _sum01 = _bias0;
                __m512 _sum02 = _bias0;
                __m512 _sum03 = _bias0;
                __m512 _sum10 = _bias0;
                __m512 _sum11 = _bias0;
                __m512 _sum12 = _bias0;
                __m512 _sum13 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);
                __m512 _r03 = _mm512_load_ps(r0 + 48);
                __m512 _r04 = _mm512_load_ps(r0 + 64);
                __m512 _r05 = _mm512_load_ps(r0 + 80);

                _sum00 = _mm512_fmadd_ps(_k00, _r00, _sum00);
                _sum01 = _mm512_fmadd_ps(_k00, _r01, _sum01);
                _sum02 = _mm512_fmadd_ps(_k00, _r02, _sum02);
                _sum03 = _mm512_fmadd_ps(_k00, _r03, _sum03);
                _sum00 = _mm512_fmadd_ps(_k01, _r01, _sum00);
                _sum01 = _mm512_fmadd_ps(_k01, _r02, _sum01);
                _sum02 = _mm512_fmadd_ps(_k01, _r03, _sum02);
                _sum03 = _mm512_fmadd_ps(_k01, _r04, _sum03);
                _sum00 = _mm512_fmadd_ps(_k02, _r02, _sum00);
                _sum01 = _mm512_fmadd_ps(_k02, _r03, _sum01);
                _sum02 = _mm512_fmadd_ps(_k02, _r04, _sum02);
                _sum03 = _mm512_fmadd_ps(_k02, _r05, _sum03);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);
                __m512 _r13 = _mm512_load_ps(r1 + 48);
                __m512 _r14 = _mm512_load_ps(r1 + 64);
                __m512 _r15 = _mm512_load_ps(r1 + 80);

                _sum10 = _mm512_fmadd_ps(_k00, _r10, _sum10);
                _sum11 = _mm512_fmadd_ps(_k00, _r11, _sum11);
                _sum12 = _mm512_fmadd_ps(_k00, _r12, _sum12);
                _sum13 = _mm512_fmadd_ps(_k00, _r13, _sum13);
                _sum00 = _mm512_fmadd_ps(_k10, _r10, _sum00);
                _sum01 = _mm512_fmadd_ps(_k10, _r11, _sum01);
                _sum02 = _mm512_fmadd_ps(_k10, _r12, _sum02);
                _sum03 = _mm512_fmadd_ps(_k10, _r13, _sum03);

                _sum10 = _mm512_fmadd_ps(_k01, _r11, _sum10);
                _sum11 = _mm512_fmadd_ps(_k01, _r12, _sum11);
                _sum12 = _mm512_fmadd_ps(_k01, _r13, _sum12);
                _sum13 = _mm512_fmadd_ps(_k01, _r14, _sum13);
                _sum00 = _mm512_fmadd_ps(_k11, _r11, _sum00);
                _sum01 = _mm512_fmadd_ps(_k11, _r12, _sum01);
                _sum02 = _mm512_fmadd_ps(_k11, _r13, _sum02);
                _sum03 = _mm512_fmadd_ps(_k11, _r14, _sum03);

                _sum10 = _mm512_fmadd_ps(_k02, _r12, _sum10);
                _sum11 = _mm512_fmadd_ps(_k02, _r13, _sum11);
                _sum12 = _mm512_fmadd_ps(_k02, _r14, _sum12);
                _sum13 = _mm512_fmadd_ps(_k02, _r15, _sum13);
                _sum00 = _mm512_fmadd_ps(_k12, _r12, _sum00);
                _sum01 = _mm512_fmadd_ps(_k12, _r13, _sum01);
                _sum02 = _mm512_fmadd_ps(_k12, _r14, _sum02);
                _sum03 = _mm512_fmadd_ps(_k12, _r15, _sum03);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);
                __m512 _r23 = _mm512_load_ps(r2 + 48);
                __m512 _r24 = _mm512_load_ps(r2 + 64);
                __m512 _r25 = _mm512_load_ps(r2 + 80);

                _sum10 = _mm512_fmadd_ps(_k10, _r20, _sum10);
                _sum11 = _mm512_fmadd_ps(_k10, _r21, _sum11);
                _sum12 = _mm512_fmadd_ps(_k10, _r22, _sum12);
                _sum13 = _mm512_fmadd_ps(_k10, _r23, _sum13);
                _sum00 = _mm512_fmadd_ps(_k20, _r20, _sum00);
                _sum01 = _mm512_fmadd_ps(_k20, _r21, _sum01);
                _sum02 = _mm512_fmadd_ps(_k20, _r22, _sum02);
                _sum03 = _mm512_fmadd_ps(_k20, _r23, _sum03);

                _sum10 = _mm512_fmadd_ps(_k11, _r21, _sum10);
                _sum11 = _mm512_fmadd_ps(_k11, _r22, _sum11);
                _sum12 = _mm512_fmadd_ps(_k11, _r23, _sum12);
                _sum13 = _mm512_fmadd_ps(_k11, _r24, _sum13);
                _sum00 = _mm512_fmadd_ps(_k21, _r21, _sum00);
                _sum01 = _mm512_fmadd_ps(_k21, _r22, _sum01);
                _sum02 = _mm512_fmadd_ps(_k21, _r23, _sum02);
                _sum03 = _mm512_fmadd_ps(_k21, _r24, _sum03);

                _sum10 = _mm512_fmadd_ps(_k12, _r22, _sum10);
                _sum11 = _mm512_fmadd_ps(_k12, _r23, _sum11);
                _sum12 = _mm512_fmadd_ps(_k12, _r24, _sum12);
                _sum13 = _mm512_fmadd_ps(_k12, _r25, _sum13);
                _sum00 = _mm512_fmadd_ps(_k22, _r22, _sum00);
                _sum01 = _mm512_fmadd_ps(_k22, _r23, _sum01);
                _sum02 = _mm512_fmadd_ps(_k22, _r24, _sum02);
                _sum03 = _mm512_fmadd_ps(_k22, _r25, _sum03);

                __m512 _r30 = _mm512_load_ps(r3);
                __m512 _r31 = _mm512_load_ps(r3 + 16);
                __m512 _r32 = _mm512_load_ps(r3 + 32);
                __m512 _r33 = _mm512_load_ps(r3 + 48);
                __m512 _r34 = _mm512_load_ps(r3 + 64);
                __m512 _r35 = _mm512_load_ps(r3 + 80);

                _sum10 = _mm512_fmadd_ps(_k20, _r30, _sum10);
                _sum11 = _mm512_fmadd_ps(_k20, _r31, _sum11);
                _sum12 = _mm512_fmadd_ps(_k20, _r32, _sum12);
                _sum13 = _mm512_fmadd_ps(_k20, _r33, _sum13);
                _sum10 = _mm512_fmadd_ps(_k21, _r31, _sum10);
                _sum11 = _mm512_fmadd_ps(_k21, _r32, _sum11);
                _sum12 = _mm512_fmadd_ps(_k21, _r33, _sum12);
                _sum13 = _mm512_fmadd_ps(_k21, _r34, _sum13);
                _sum10 = _mm512_fmadd_ps(_k22, _r32, _sum10);
                _sum11 = _mm512_fmadd_ps(_k22, _r33, _sum11);
                _sum12 = _mm512_fmadd_ps(_k22, _r34, _sum12);
                _sum13 = _mm512_fmadd_ps(_k22, _r35, _sum13);

                _mm512_store_ps(outptr0, _sum00);
                _mm512_store_ps(outptr0 + 16, _sum01);
                _mm512_store_ps(outptr0 + 32, _sum02);
                _mm512_store_ps(outptr0 + 48, _sum03);
                _mm512_store_ps(outptr1, _sum10);
                _mm512_store_ps(outptr1 + 16, _sum11);
                _mm512_store_ps(outptr1 + 32, _sum12);
                _mm512_store_ps(outptr1 + 48, _sum13);

                r0 += 64;
                r1 += 64;
                r2 += 64;
                r3 += 64;
                outptr0 += 64;
                outptr1 += 64;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m512 _sum00 = _bias0;
                __m512 _sum01 = _bias0;
                __m512 _sum10 = _bias0;
                __m512 _sum11 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);
                __m512 _r03 = _mm512_load_ps(r0 + 48);

                _sum00 = _mm512_fmadd_ps(_k00, _r00, _sum00);
                _sum01 = _mm512_fmadd_ps(_k00, _r01, _sum01);
                _sum00 = _mm512_fmadd_ps(_k01, _r01, _sum00);
                _sum01 = _mm512_fmadd_ps(_k01, _r02, _sum01);
                _sum00 = _mm512_fmadd_ps(_k02, _r02, _sum00);
                _sum01 = _mm512_fmadd_ps(_k02, _r03, _sum01);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);
                __m512 _r13 = _mm512_load_ps(r1 + 48);

                _sum00 = _mm512_fmadd_ps(_k10, _r10, _sum00);
                _sum01 = _mm512_fmadd_ps(_k10, _r11, _sum01);
                _sum10 = _mm512_fmadd_ps(_k00, _r10, _sum10);
                _sum11 = _mm512_fmadd_ps(_k00, _r11, _sum11);

                _sum00 = _mm512_fmadd_ps(_k11, _r11, _sum00);
                _sum01 = _mm512_fmadd_ps(_k11, _r12, _sum01);
                _sum10 = _mm512_fmadd_ps(_k01, _r11, _sum10);
                _sum11 = _mm512_fmadd_ps(_k01, _r12, _sum11);

                _sum00 = _mm512_fmadd_ps(_k12, _r12, _sum00);
                _sum01 = _mm512_fmadd_ps(_k12, _r13, _sum01);
                _sum10 = _mm512_fmadd_ps(_k02, _r12, _sum10);
                _sum11 = _mm512_fmadd_ps(_k02, _r13, _sum11);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);
                __m512 _r23 = _mm512_load_ps(r2 + 48);

                _sum00 = _mm512_fmadd_ps(_k20, _r20, _sum00);
                _sum01 = _mm512_fmadd_ps(_k20, _r21, _sum01);
                _sum10 = _mm512_fmadd_ps(_k10, _r20, _sum10);
                _sum11 = _mm512_fmadd_ps(_k10, _r21, _sum11);

                _sum00 = _mm512_fmadd_ps(_k21, _r21, _sum00);
                _sum01 = _mm512_fmadd_ps(_k21, _r22, _sum01);
                _sum10 = _mm512_fmadd_ps(_k11, _r21, _sum10);
                _sum11 = _mm512_fmadd_ps(_k11, _r22, _sum11);

                _sum00 = _mm512_fmadd_ps(_k22, _r22, _sum00);
                _sum01 = _mm512_fmadd_ps(_k22, _r23, _sum01);
                _sum10 = _mm512_fmadd_ps(_k12, _r22, _sum10);
                _sum11 = _mm512_fmadd_ps(_k12, _r23, _sum11);

                __m512 _r30 = _mm512_load_ps(r3);
                __m512 _r31 = _mm512_load_ps(r3 + 16);
                __m512 _r32 = _mm512_load_ps(r3 + 32);
                __m512 _r33 = _mm512_load_ps(r3 + 48);

                _sum10 = _mm512_fmadd_ps(_k20, _r30, _sum10);
                _sum11 = _mm512_fmadd_ps(_k20, _r31, _sum11);
                _sum10 = _mm512_fmadd_ps(_k21, _r31, _sum10);
                _sum11 = _mm512_fmadd_ps(_k21, _r32, _sum11);
                _sum10 = _mm512_fmadd_ps(_k22, _r32, _sum10);
                _sum11 = _mm512_fmadd_ps(_k22, _r33, _sum11);

                _mm512_store_ps(outptr0, _sum00);
                _mm512_store_ps(outptr0 + 16, _sum01);
                _mm512_store_ps(outptr1, _sum10);
                _mm512_store_ps(outptr1 + 16, _sum11);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                r3 += 32;
                outptr0 += 32;
                outptr1 += 32;
            }
            for (; j < outw; j++)
            {
                __m512 _sum0 = _bias0;
                __m512 _sum1 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm512_fmadd_ps(_k00, _r10, _sum1);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm512_fmadd_ps(_k01, _r11, _sum1);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm512_fmadd_ps(_k02, _r12, _sum1);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm512_fmadd_ps(_k10, _r20, _sum1);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm512_fmadd_ps(_k11, _r21, _sum1);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm512_fmadd_ps(_k12, _r22, _sum1);

                __m512 _r30 = _mm512_load_ps(r3);
                __m512 _r31 = _mm512_load_ps(r3 + 16);
                __m512 _r32 = _mm512_load_ps(r3 + 32);

                _sum1 = _mm512_fmadd_ps(_k20, _r30, _sum1);
                _sum1 = _mm512_fmadd_ps(_k21, _r31, _sum1);
                _sum1 = _mm512_fmadd_ps(_k22, _r32, _sum1);

                _mm512_store_ps(outptr0, _sum0);
                _mm512_store_ps(outptr1, _sum1);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                outptr0 += 16;
                outptr1 += 16;
            }

            r0 += 2 * 16 + w * 16;
            r1 += 2 * 16 + w * 16;
            r2 += 2 * 16 + w * 16;
            r3 += 2 * 16 + w * 16;

            outptr0 += outw * 16;
            outptr1 += outw * 16;
        }
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m512 _sum0 = _bias0;
                __m512 _sum1 = _bias0;
                __m512 _sum2 = _bias0;
                __m512 _sum3 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);
                __m512 _r03 = _mm512_load_ps(r0 + 48);
                __m512 _r04 = _mm512_load_ps(r0 + 64);
                __m512 _r05 = _mm512_load_ps(r0 + 80);

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum1 = _mm512_fmadd_ps(_k00, _r01, _sum1);
                _sum2 = _mm512_fmadd_ps(_k00, _r02, _sum2);
                _sum3 = _mm512_fmadd_ps(_k00, _r03, _sum3);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum1 = _mm512_fmadd_ps(_k01, _r02, _sum1);
                _sum2 = _mm512_fmadd_ps(_k01, _r03, _sum2);
                _sum3 = _mm512_fmadd_ps(_k01, _r04, _sum3);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);
                _sum1 = _mm512_fmadd_ps(_k02, _r03, _sum1);
                _sum2 = _mm512_fmadd_ps(_k02, _r04, _sum2);
                _sum3 = _mm512_fmadd_ps(_k02, _r05, _sum3);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);
                __m512 _r13 = _mm512_load_ps(r1 + 48);
                __m512 _r14 = _mm512_load_ps(r1 + 64);
                __m512 _r15 = _mm512_load_ps(r1 + 80);

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm512_fmadd_ps(_k10, _r11, _sum1);
                _sum2 = _mm512_fmadd_ps(_k10, _r12, _sum2);
                _sum3 = _mm512_fmadd_ps(_k10, _r13, _sum3);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm512_fmadd_ps(_k11, _r12, _sum1);
                _sum2 = _mm512_fmadd_ps(_k11, _r13, _sum2);
                _sum3 = _mm512_fmadd_ps(_k11, _r14, _sum3);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm512_fmadd_ps(_k12, _r13, _sum1);
                _sum2 = _mm512_fmadd_ps(_k12, _r14, _sum2);
                _sum3 = _mm512_fmadd_ps(_k12, _r15, _sum3);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);
                __m512 _r23 = _mm512_load_ps(r2 + 48);
                __m512 _r24 = _mm512_load_ps(r2 + 64);
                __m512 _r25 = _mm512_load_ps(r2 + 80);

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm512_fmadd_ps(_k20, _r21, _sum1);
                _sum2 = _mm512_fmadd_ps(_k20, _r22, _sum2);
                _sum3 = _mm512_fmadd_ps(_k20, _r23, _sum3);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm512_fmadd_ps(_k21, _r22, _sum1);
                _sum2 = _mm512_fmadd_ps(_k21, _r23, _sum2);
                _sum3 = _mm512_fmadd_ps(_k21, _r24, _sum3);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm512_fmadd_ps(_k22, _r23, _sum1);
                _sum2 = _mm512_fmadd_ps(_k22, _r24, _sum2);
                _sum3 = _mm512_fmadd_ps(_k22, _r25, _sum3);

                _mm512_store_ps(outptr0, _sum0);
                _mm512_store_ps(outptr0 + 16, _sum1);
                _mm512_store_ps(outptr0 + 32, _sum2);
                _mm512_store_ps(outptr0 + 48, _sum3);

                r0 += 64;
                r1 += 64;
                r2 += 64;
                outptr0 += 64;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m512 _sum0 = _bias0;
                __m512 _sum1 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);
                __m512 _r03 = _mm512_load_ps(r0 + 48);

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum1 = _mm512_fmadd_ps(_k00, _r01, _sum1);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum1 = _mm512_fmadd_ps(_k01, _r02, _sum1);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);
                _sum1 = _mm512_fmadd_ps(_k02, _r03, _sum1);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);
                __m512 _r13 = _mm512_load_ps(r1 + 48);

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm512_fmadd_ps(_k10, _r11, _sum1);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm512_fmadd_ps(_k11, _r12, _sum1);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm512_fmadd_ps(_k12, _r13, _sum1);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);
                __m512 _r23 = _mm512_load_ps(r2 + 48);

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm512_fmadd_ps(_k20, _r21, _sum1);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm512_fmadd_ps(_k21, _r22, _sum1);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm512_fmadd_ps(_k22, _r23, _sum1);

                _mm512_store_ps(outptr0, _sum0);
                _mm512_store_ps(outptr0 + 16, _sum1);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                outptr0 += 32;
            }
            for (; j < outw; j++)
            {
                __m512 _sum0 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);

                _mm512_store_ps(outptr0, _sum0);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr0 += 16;
            }

            r0 += 2 * 16;
            r1 += 2 * 16;
            r2 += 2 * 16;
        }
    }
}

static void convdw3x3s2_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * 16;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __m512 _bias0 = bias ? _mm512_loadu_ps((const float*)bias + g * 16) : _mm512_setzero_ps();

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        __m512 _k00 = _mm512_load_ps(k0);
        __m512 _k01 = _mm512_load_ps(k0 + 16);
        __m512 _k02 = _mm512_load_ps(k0 + 32);
        __m512 _k10 = _mm512_load_ps(k0 + 48);
        __m512 _k11 = _mm512_load_ps(k0 + 64);
        __m512 _k12 = _mm512_load_ps(k0 + 80);
        __m512 _k20 = _mm512_load_ps(k0 + 96);
        __m512 _k21 = _mm512_load_ps(k0 + 112);
        __m512 _k22 = _mm512_load_ps(k0 + 128);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m512 _sum0 = _bias0;
                __m512 _sum1 = _bias0;
                __m512 _sum2 = _bias0;
                __m512 _sum3 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);
                __m512 _r03 = _mm512_load_ps(r0 + 48);
                __m512 _r04 = _mm512_load_ps(r0 + 64);
                __m512 _r05 = _mm512_load_ps(r0 + 80);
                __m512 _r06 = _mm512_load_ps(r0 + 96);
                __m512 _r07 = _mm512_load_ps(r0 + 112);
                __m512 _r08 = _mm512_load_ps(r0 + 128);

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum1 = _mm512_fmadd_ps(_k00, _r02, _sum1);
                _sum2 = _mm512_fmadd_ps(_k00, _r04, _sum2);
                _sum3 = _mm512_fmadd_ps(_k00, _r06, _sum3);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum1 = _mm512_fmadd_ps(_k01, _r03, _sum1);
                _sum2 = _mm512_fmadd_ps(_k01, _r05, _sum2);
                _sum3 = _mm512_fmadd_ps(_k01, _r07, _sum3);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);
                _sum1 = _mm512_fmadd_ps(_k02, _r04, _sum1);
                _sum2 = _mm512_fmadd_ps(_k02, _r06, _sum2);
                _sum3 = _mm512_fmadd_ps(_k02, _r08, _sum3);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);
                __m512 _r13 = _mm512_load_ps(r1 + 48);
                __m512 _r14 = _mm512_load_ps(r1 + 64);
                __m512 _r15 = _mm512_load_ps(r1 + 80);
                __m512 _r16 = _mm512_load_ps(r1 + 96);
                __m512 _r17 = _mm512_load_ps(r1 + 112);
                __m512 _r18 = _mm512_load_ps(r1 + 128);

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm512_fmadd_ps(_k10, _r12, _sum1);
                _sum2 = _mm512_fmadd_ps(_k10, _r14, _sum2);
                _sum3 = _mm512_fmadd_ps(_k10, _r16, _sum3);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm512_fmadd_ps(_k11, _r13, _sum1);
                _sum2 = _mm512_fmadd_ps(_k11, _r15, _sum2);
                _sum3 = _mm512_fmadd_ps(_k11, _r17, _sum3);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm512_fmadd_ps(_k12, _r14, _sum1);
                _sum2 = _mm512_fmadd_ps(_k12, _r16, _sum2);
                _sum3 = _mm512_fmadd_ps(_k12, _r18, _sum3);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);
                __m512 _r23 = _mm512_load_ps(r2 + 48);
                __m512 _r24 = _mm512_load_ps(r2 + 64);
                __m512 _r25 = _mm512_load_ps(r2 + 80);
                __m512 _r26 = _mm512_load_ps(r2 + 96);
                __m512 _r27 = _mm512_load_ps(r2 + 112);
                __m512 _r28 = _mm512_load_ps(r2 + 128);

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm512_fmadd_ps(_k20, _r22, _sum1);
                _sum2 = _mm512_fmadd_ps(_k20, _r24, _sum2);
                _sum3 = _mm512_fmadd_ps(_k20, _r26, _sum3);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm512_fmadd_ps(_k21, _r23, _sum1);
                _sum2 = _mm512_fmadd_ps(_k21, _r25, _sum2);
                _sum3 = _mm512_fmadd_ps(_k21, _r27, _sum3);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm512_fmadd_ps(_k22, _r24, _sum1);
                _sum2 = _mm512_fmadd_ps(_k22, _r26, _sum2);
                _sum3 = _mm512_fmadd_ps(_k22, _r28, _sum3);

                _mm512_store_ps(outptr0, _sum0);
                _mm512_store_ps(outptr0 + 16, _sum1);
                _mm512_store_ps(outptr0 + 32, _sum2);
                _mm512_store_ps(outptr0 + 48, _sum3);

                r0 += 2 * 64;
                r1 += 2 * 64;
                r2 += 2 * 64;
                outptr0 += 64;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m512 _sum0 = _bias0;
                __m512 _sum1 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);
                __m512 _r03 = _mm512_load_ps(r0 + 48);
                __m512 _r04 = _mm512_load_ps(r0 + 64);

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum1 = _mm512_fmadd_ps(_k00, _r02, _sum1);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum1 = _mm512_fmadd_ps(_k01, _r03, _sum1);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);
                _sum1 = _mm512_fmadd_ps(_k02, _r04, _sum1);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);
                __m512 _r13 = _mm512_load_ps(r1 + 48);
                __m512 _r14 = _mm512_load_ps(r1 + 64);

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm512_fmadd_ps(_k10, _r12, _sum1);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm512_fmadd_ps(_k11, _r13, _sum1);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm512_fmadd_ps(_k12, _r14, _sum1);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);
                __m512 _r23 = _mm512_load_ps(r2 + 48);
                __m512 _r24 = _mm512_load_ps(r2 + 64);

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm512_fmadd_ps(_k20, _r22, _sum1);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm512_fmadd_ps(_k21, _r23, _sum1);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm512_fmadd_ps(_k22, _r24, _sum1);

                _mm512_store_ps(outptr0, _sum0);
                _mm512_store_ps(outptr0 + 16, _sum1);

                r0 += 2 * 32;
                r1 += 2 * 32;
                r2 += 2 * 32;
                outptr0 += 32;
            }
            for (; j < outw; j++)
            {
                __m512 _sum0 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);

                _mm512_store_ps(outptr0, _sum0);

                r0 += 2 * 16;
                r1 += 2 * 16;
                r2 += 2 * 16;
                outptr0 += 16;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

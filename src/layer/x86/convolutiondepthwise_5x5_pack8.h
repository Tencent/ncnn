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

static void convdw5x5s1_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + g * 8) : _mm256_set1_ps(0.f);

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);
        float* outptr1 = out.row(1);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        const float* r3 = img0.row(3);
        const float* r4 = img0.row(4);
        const float* r5 = img0.row(5);

        int i = 0;
#if __aarch64__
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;

            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum00 = _bias0;
                __m256 _sum01 = _bias0;
                __m256 _sum02 = _bias0;
                __m256 _sum03 = _bias0;
                __m256 _sum10 = _bias0;
                __m256 _sum11 = _bias0;
                __m256 _sum12 = _bias0;
                __m256 _sum13 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 8);
                __m256 _r02 = _mm256_loadu_ps(r0 + 16);
                __m256 _r03 = _mm256_loadu_ps(r0 + 24);
                __m256 _r04 = _mm256_loadu_ps(r0 + 32);
                __m256 _r05 = _mm256_loadu_ps(r0 + 40);
                __m256 _r06 = _mm256_loadu_ps(r0 + 48);
                __m256 _r07 = _mm256_loadu_ps(r0 + 56);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 8);
                __m256 _k02 = _mm256_loadu_ps(k0 + 16);
                __m256 _k03 = _mm256_loadu_ps(k0 + 24);
                __m256 _k04 = _mm256_loadu_ps(k0 + 32);
                k0 += 40;

                _sum00 = _mm256_fmadd_ps(_k00, _r00,_sum00);
                _sum00 = _mm256_fmadd_ps(_k01, _r01,_sum00);
                _sum00 = _mm256_fmadd_ps(_k02, _r02,_sum00);
                _sum00 = _mm256_fmadd_ps(_k03, _r03,_sum00);
                _sum00 = _mm256_fmadd_ps(_k04, _r04,_sum00);
                _sum01 = _mm256_fmadd_ps(_k00, _r01,_sum01);
                _sum01 = _mm256_fmadd_ps(_k01, _r02,_sum01);
                _sum01 = _mm256_fmadd_ps(_k02, _r03,_sum01);
                _sum01 = _mm256_fmadd_ps(_k03, _r04,_sum01);
                _sum01 = _mm256_fmadd_ps(_k04, _r05,_sum01);
                _sum02 = _mm256_fmadd_ps(_k00, _r02,_sum02);
                _sum02 = _mm256_fmadd_ps(_k01, _r03,_sum02);
                _sum02 = _mm256_fmadd_ps(_k02, _r04,_sum02);
                _sum02 = _mm256_fmadd_ps(_k03, _r05,_sum02);
                _sum02 = _mm256_fmadd_ps(_k04, _r06,_sum02);
                _sum03 = _mm256_fmadd_ps(_k00, _r03,_sum03);
                _sum03 = _mm256_fmadd_ps(_k01, _r04,_sum03);
                _sum03 = _mm256_fmadd_ps(_k02, _r05,_sum03);
                _sum03 = _mm256_fmadd_ps(_k03, _r06,_sum03);
                _sum03 = _mm256_fmadd_ps(_k04, _r07,_sum03);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 8);
                __m256 _r12 = _mm256_loadu_ps(r1 + 16);
                __m256 _r13 = _mm256_loadu_ps(r1 + 24);
                __m256 _r14 = _mm256_loadu_ps(r1 + 32);
                __m256 _r15 = _mm256_loadu_ps(r1 + 40);
                __m256 _r16 = _mm256_loadu_ps(r1 + 48);
                __m256 _r17 = _mm256_loadu_ps(r1 + 56);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 8);
                __m256 _k12 = _mm256_loadu_ps(k0 + 16);
                __m256 _k13 = _mm256_loadu_ps(k0 + 24);
                __m256 _k14 = _mm256_loadu_ps(k0 + 32);
                k0 += 40;

                _sum10 = _mm256_fmadd_ps(_k00, _r10,_sum10);
                _sum10 = _mm256_fmadd_ps(_k01, _r11,_sum10);
                _sum10 = _mm256_fmadd_ps(_k02, _r12,_sum10);
                _sum10 = _mm256_fmadd_ps(_k03, _r13,_sum10);
                _sum10 = _mm256_fmadd_ps(_k04, _r14,_sum10);
                _sum11 = _mm256_fmadd_ps(_k00, _r11,_sum11);
                _sum11 = _mm256_fmadd_ps(_k01, _r12,_sum11);
                _sum11 = _mm256_fmadd_ps(_k02, _r13,_sum11);
                _sum11 = _mm256_fmadd_ps(_k03, _r14,_sum11);
                _sum11 = _mm256_fmadd_ps(_k04, _r15,_sum11);
                _sum12 = _mm256_fmadd_ps(_k00, _r12,_sum12);
                _sum12 = _mm256_fmadd_ps(_k01, _r13,_sum12);
                _sum12 = _mm256_fmadd_ps(_k02, _r14,_sum12);
                _sum12 = _mm256_fmadd_ps(_k03, _r15,_sum12);
                _sum12 = _mm256_fmadd_ps(_k04, _r16,_sum12);
                _sum13 = _mm256_fmadd_ps(_k00, _r13,_sum13);
                _sum13 = _mm256_fmadd_ps(_k01, _r14,_sum13);
                _sum13 = _mm256_fmadd_ps(_k02, _r15,_sum13);
                _sum13 = _mm256_fmadd_ps(_k03, _r16,_sum13);
                _sum13 = _mm256_fmadd_ps(_k04, _r17,_sum13);

                _sum00 = _mm256_fmadd_ps( _k10, _r10,_sum00);
                _sum00 = _mm256_fmadd_ps( _k11, _r11,_sum00);
                _sum00 = _mm256_fmadd_ps( _k12, _r12,_sum00);
                _sum00 = _mm256_fmadd_ps( _k13, _r13,_sum00);
                _sum00 = _mm256_fmadd_ps( _k14, _r14,_sum00);
                _sum01 = _mm256_fmadd_ps( _k10, _r11,_sum01);
                _sum01 = _mm256_fmadd_ps( _k11, _r12,_sum01);
                _sum01 = _mm256_fmadd_ps( _k12, _r13,_sum01);
                _sum01 = _mm256_fmadd_ps( _k13, _r14,_sum01);
                _sum01 = _mm256_fmadd_ps( _k14, _r15,_sum01);
                _sum02 = _mm256_fmadd_ps( _k10, _r12,_sum02);
                _sum02 = _mm256_fmadd_ps( _k11, _r13,_sum02);
                _sum02 = _mm256_fmadd_ps( _k12, _r14,_sum02);
                _sum02 = _mm256_fmadd_ps( _k13, _r15,_sum02);
                _sum02 = _mm256_fmadd_ps( _k14, _r16,_sum02);
                _sum03 = _mm256_fmadd_ps( _k10, _r13,_sum03);
                _sum03 = _mm256_fmadd_ps( _k11, _r14,_sum03);
                _sum03 = _mm256_fmadd_ps( _k12, _r15,_sum03);
                _sum03 = _mm256_fmadd_ps( _k13, _r16,_sum03);
                _sum03 = _mm256_fmadd_ps( _k14, _r17,_sum03);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 8);
                __m256 _r22 = _mm256_loadu_ps(r2 + 16);
                __m256 _r23 = _mm256_loadu_ps(r2 + 24);
                __m256 _r24 = _mm256_loadu_ps(r2 + 32);
                __m256 _r25 = _mm256_loadu_ps(r2 + 40);
                __m256 _r26 = _mm256_loadu_ps(r2 + 48);
                __m256 _r27 = _mm256_loadu_ps(r2 + 56);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 8);
                __m256 _k22 = _mm256_loadu_ps(k0 + 16);
                __m256 _k23 = _mm256_loadu_ps(k0 + 24);
                __m256 _k24 = _mm256_loadu_ps(k0 + 32);
                k0 += 40;

                _sum10 = _mm256_fmadd_ps( _k10, _r20,_sum10);
                _sum10 = _mm256_fmadd_ps( _k11, _r21,_sum10);
                _sum10 = _mm256_fmadd_ps( _k12, _r22,_sum10);
                _sum10 = _mm256_fmadd_ps( _k13, _r23,_sum10);
                _sum10 = _mm256_fmadd_ps( _k14, _r24,_sum10);
                _sum11 = _mm256_fmadd_ps( _k10, _r21,_sum11);
                _sum11 = _mm256_fmadd_ps( _k11, _r22,_sum11);
                _sum11 = _mm256_fmadd_ps( _k12, _r23,_sum11);
                _sum11 = _mm256_fmadd_ps( _k13, _r24,_sum11);
                _sum11 = _mm256_fmadd_ps( _k14, _r25,_sum11);
                _sum12 = _mm256_fmadd_ps( _k10, _r22,_sum12);
                _sum12 = _mm256_fmadd_ps( _k11, _r23,_sum12);
                _sum12 = _mm256_fmadd_ps( _k12, _r24,_sum12);
                _sum12 = _mm256_fmadd_ps( _k13, _r25,_sum12);
                _sum12 = _mm256_fmadd_ps( _k14, _r26,_sum12);
                _sum13 = _mm256_fmadd_ps( _k10, _r23,_sum13);
                _sum13 = _mm256_fmadd_ps( _k11, _r24,_sum13);
                _sum13 = _mm256_fmadd_ps( _k12, _r25,_sum13);
                _sum13 = _mm256_fmadd_ps( _k13, _r26,_sum13);
                _sum13 = _mm256_fmadd_ps( _k14, _r27,_sum13);

                _sum00 = _mm256_fmadd_ps( _k20, _r20,_sum00s);
                _sum00 = _mm256_fmadd_ps( _k21, _r21,_sum00s);
                _sum00 = _mm256_fmadd_ps( _k22, _r22,_sum00s);
                _sum00 = _mm256_fmadd_ps( _k23, _r23,_sum00s);
                _sum00 = _mm256_fmadd_ps( _k24, _r24,_sum00s);
                _sum01 = _mm256_fmadd_ps( _k20, _r21,_sum01s);
                _sum01 = _mm256_fmadd_ps( _k21, _r22,_sum01s);
                _sum01 = _mm256_fmadd_ps( _k22, _r23,_sum01s);
                _sum01 = _mm256_fmadd_ps( _k23, _r24,_sum01s);
                _sum01 = _mm256_fmadd_ps( _k24, _r25,_sum01s);
                _sum02 = _mm256_fmadd_ps( _k20, _r22,_sum02s);
                _sum02 = _mm256_fmadd_ps( _k21, _r23,_sum02s);
                _sum02 = _mm256_fmadd_ps( _k22, _r24,_sum02s);
                _sum02 = _mm256_fmadd_ps( _k23, _r25,_sum02s);
                _sum02 = _mm256_fmadd_ps( _k24, _r26,_sum02s);
                _sum03 = _mm256_fmadd_ps( _k20, _r23,_sum03s);
                _sum03 = _mm256_fmadd_ps( _k21, _r24,_sum03s);
                _sum03 = _mm256_fmadd_ps( _k22, _r25,_sum03s);
                _sum03 = _mm256_fmadd_ps( _k23, _r26,_sum03s);
                _sum03 = _mm256_fmadd_ps( _k24, _r27,_sum03s);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 8);
                __m256 _r32 = _mm256_loadu_ps(r3 + 16);
                __m256 _r33 = _mm256_loadu_ps(r3 + 24);
                __m256 _r34 = _mm256_loadu_ps(r3 + 32);
                __m256 _r35 = _mm256_loadu_ps(r3 + 40);
                __m256 _r36 = _mm256_loadu_ps(r3 + 48);
                __m256 _r37 = _mm256_loadu_ps(r3 + 56);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 8);
                __m256 _k32 = _mm256_loadu_ps(k0 + 16);
                __m256 _k33 = _mm256_loadu_ps(k0 + 24);
                __m256 _k34 = _mm256_loadu_ps(k0 + 32);
                k0 += 40;

                _sum10 = _mm256_fmadd_ps( _k20, _r30,_sum10);
                _sum10 = _mm256_fmadd_ps( _k21, _r31,_sum10);
                _sum10 = _mm256_fmadd_ps( _k22, _r32,_sum10);
                _sum10 = _mm256_fmadd_ps( _k23, _r33,_sum10);
                _sum10 = _mm256_fmadd_ps( _k24, _r34,_sum10);
                _sum11 = _mm256_fmadd_ps( _k20, _r31,_sum11);
                _sum11 = _mm256_fmadd_ps( _k21, _r32,_sum11);
                _sum11 = _mm256_fmadd_ps( _k22, _r33,_sum11);
                _sum11 = _mm256_fmadd_ps( _k23, _r34,_sum11);
                _sum11 = _mm256_fmadd_ps( _k24, _r35,_sum11);
                _sum12 = _mm256_fmadd_ps( _k20, _r32,_sum12);
                _sum12 = _mm256_fmadd_ps( _k21, _r33,_sum12);
                _sum12 = _mm256_fmadd_ps( _k22, _r34,_sum12);
                _sum12 = _mm256_fmadd_ps( _k23, _r35,_sum12);
                _sum12 = _mm256_fmadd_ps( _k24, _r36,_sum12);
                _sum13 = _mm256_fmadd_ps( _k20, _r33,_sum13);
                _sum13 = _mm256_fmadd_ps( _k21, _r34,_sum13);
                _sum13 = _mm256_fmadd_ps( _k22, _r35,_sum13);
                _sum13 = _mm256_fmadd_ps( _k23, _r36,_sum13);
                _sum13 = _mm256_fmadd_ps( _k24, _r37,_sum13);

                _sum00 = _mm256_fmadd_ps(_k30, _r30,_sum00);
                _sum00 = _mm256_fmadd_ps(_k31, _r31,_sum00);
                _sum00 = _mm256_fmadd_ps(_k32, _r32,_sum00);
                _sum00 = _mm256_fmadd_ps(_k33, _r33,_sum00);
                _sum00 = _mm256_fmadd_ps(_k34, _r34,_sum00);
                _sum01 = _mm256_fmadd_ps(_k30, _r31,_sum01);
                _sum01 = _mm256_fmadd_ps(_k31, _r32,_sum01);
                _sum01 = _mm256_fmadd_ps(_k32, _r33,_sum01);
                _sum01 = _mm256_fmadd_ps(_k33, _r34,_sum01);
                _sum01 = _mm256_fmadd_ps(_k34, _r35,_sum01);
                _sum02 = _mm256_fmadd_ps(_k30, _r32,_sum02);
                _sum02 = _mm256_fmadd_ps(_k31, _r33,_sum02);
                _sum02 = _mm256_fmadd_ps(_k32, _r34,_sum02);
                _sum02 = _mm256_fmadd_ps(_k33, _r35,_sum02);
                _sum02 = _mm256_fmadd_ps(_k34, _r36,_sum02);
                _sum03 = _mm256_fmadd_ps(_k30, _r33,_sum03);
                _sum03 = _mm256_fmadd_ps(_k31, _r34,_sum03);
                _sum03 = _mm256_fmadd_ps(_k32, _r35,_sum03);
                _sum03 = _mm256_fmadd_ps(_k33, _r36,_sum03);
                _sum03 = _mm256_fmadd_ps(_k34, _r37,_sum03);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 8);
                __m256 _r42 = _mm256_loadu_ps(r4 + 16);
                __m256 _r43 = _mm256_loadu_ps(r4 + 24);
                __m256 _r44 = _mm256_loadu_ps(r4 + 32);
                __m256 _r45 = _mm256_loadu_ps(r4 + 40);
                __m256 _r46 = _mm256_loadu_ps(r4 + 48);
                __m256 _r47 = _mm256_loadu_ps(r4 + 56);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 8);
                __m256 _k42 = _mm256_loadu_ps(k0 + 16);
                __m256 _k43 = _mm256_loadu_ps(k0 + 24);
                __m256 _k44 = _mm256_loadu_ps(k0 + 32);
                k0 -= 160;

                _sum10 = _mm256_fmadd_ps(_k30, _r40,_sum10);
                _sum10 = _mm256_fmadd_ps(_k31, _r41,_sum10);
                _sum10 = _mm256_fmadd_ps(_k32, _r42,_sum10);
                _sum10 = _mm256_fmadd_ps(_k33, _r43,_sum10);
                _sum10 = _mm256_fmadd_ps(_k34, _r44,_sum10);
                _sum11 = _mm256_fmadd_ps(_k30, _r41,_sum11);
                _sum11 = _mm256_fmadd_ps(_k31, _r42,_sum11);
                _sum11 = _mm256_fmadd_ps(_k32, _r43,_sum11);
                _sum11 = _mm256_fmadd_ps(_k33, _r44,_sum11);
                _sum11 = _mm256_fmadd_ps(_k34, _r45,_sum11);
                _sum12 = _mm256_fmadd_ps(_k30, _r42,_sum12);
                _sum12 = _mm256_fmadd_ps(_k31, _r43,_sum12);
                _sum12 = _mm256_fmadd_ps(_k32, _r44,_sum12);
                _sum12 = _mm256_fmadd_ps(_k33, _r45,_sum12);
                _sum12 = _mm256_fmadd_ps(_k34, _r46,_sum12);
                _sum13 = _mm256_fmadd_ps(_k30, _r43,_sum13);
                _sum13 = _mm256_fmadd_ps(_k31, _r44,_sum13);
                _sum13 = _mm256_fmadd_ps(_k32, _r45,_sum13);
                _sum13 = _mm256_fmadd_ps(_k33, _r46,_sum13);
                _sum13 = _mm256_fmadd_ps(_k34, _r47,_sum13);

                _sum00 = _mm256_fmadd_ps(_k40, _r40,_sum00);
                _sum00 = _mm256_fmadd_ps(_k41, _r41,_sum00);
                _sum00 = _mm256_fmadd_ps(_k42, _r42,_sum00);
                _sum00 = _mm256_fmadd_ps(_k43, _r43,_sum00);
                _sum00 = _mm256_fmadd_ps(_k44, _r44,_sum00);
                _sum01 = _mm256_fmadd_ps(_k40, _r41,_sum01);
                _sum01 = _mm256_fmadd_ps(_k41, _r42,_sum01);
                _sum01 = _mm256_fmadd_ps(_k42, _r43,_sum01);
                _sum01 = _mm256_fmadd_ps(_k43, _r44,_sum01);
                _sum01 = _mm256_fmadd_ps(_k44, _r45,_sum01);
                _sum02 = _mm256_fmadd_ps(_k40, _r42,_sum02);
                _sum02 = _mm256_fmadd_ps(_k41, _r43,_sum02);
                _sum02 = _mm256_fmadd_ps(_k42, _r44,_sum02);
                _sum02 = _mm256_fmadd_ps(_k43, _r45,_sum02);
                _sum02 = _mm256_fmadd_ps(_k44, _r46,_sum02);
                _sum03 = _mm256_fmadd_ps(_k40, _r43,_sum03);
                _sum03 = _mm256_fmadd_ps(_k41, _r44,_sum03);
                _sum03 = _mm256_fmadd_ps(_k42, _r45,_sum03);
                _sum03 = _mm256_fmadd_ps(_k43, _r46,_sum03);
                _sum03 = _mm256_fmadd_ps(_k44, _r47,_sum03);

                __m256 _r50 = _mm256_loadu_ps(r5);
                __m256 _r51 = _mm256_loadu_ps(r5 + 8);
                __m256 _r52 = _mm256_loadu_ps(r5 + 16);
                __m256 _r53 = _mm256_loadu_ps(r5 + 24);
                __m256 _r54 = _mm256_loadu_ps(r5 + 32);
                __m256 _r55 = _mm256_loadu_ps(r5 + 40);
                __m256 _r56 = _mm256_loadu_ps(r5 + 48);
                __m256 _r57 = _mm256_loadu_ps(r5 + 56);

                _sum10 = _mm256_fmadd_ps(_k40, _r50,_sum10);
                _sum10 = _mm256_fmadd_ps(_k41, _r51,_sum10);
                _sum10 = _mm256_fmadd_ps(_k42, _r52,_sum10);
                _sum10 = _mm256_fmadd_ps(_k43, _r53,_sum10);
                _sum10 = _mm256_fmadd_ps(_k44, _r54,_sum10);
                _sum11 = _mm256_fmadd_ps(_k40, _r51,_sum11);
                _sum11 = _mm256_fmadd_ps(_k41, _r52,_sum11);
                _sum11 = _mm256_fmadd_ps(_k42, _r53,_sum11);
                _sum11 = _mm256_fmadd_ps(_k43, _r54,_sum11);
                _sum11 = _mm256_fmadd_ps(_k44, _r55,_sum11);
                _sum12 = _mm256_fmadd_ps(_k40, _r52,_sum12);
                _sum12 = _mm256_fmadd_ps(_k41, _r53,_sum12);
                _sum12 = _mm256_fmadd_ps(_k42, _r54,_sum12);
                _sum12 = _mm256_fmadd_ps(_k43, _r55,_sum12);
                _sum12 = _mm256_fmadd_ps(_k44, _r56,_sum12);
                _sum13 = _mm256_fmadd_ps(_k40, _r53,_sum13);
                _sum13 = _mm256_fmadd_ps(_k41, _r54,_sum13);
                _sum13 = _mm256_fmadd_ps(_k42, _r55,_sum13);
                _sum13 = _mm256_fmadd_ps(_k43, _r56,_sum13);
                _sum13 = _mm256_fmadd_ps(_k44, _r57,_sum13);

                _mm256_storeu_ps(outptr0, _sum00);
                _mm256_storeu_ps(outptr0 + 4, _sum01);
                _mm256_storeu_ps(outptr0 + 8, _sum02);
                _mm256_storeu_ps(outptr0 + 12, _sum03);
                _mm256_storeu_ps(outptr1, _sum10);
                _mm256_storeu_ps(outptr1 + 4, _sum11);
                _mm256_storeu_ps(outptr1 + 8, _sum12);
                _mm256_storeu_ps(outptr1 + 12, _sum13);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                r4 += 16;
                r5 += 16;
                outptr0 += 16;
                outptr1 += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum00 = _bias0;
                __m256 _sum01 = _bias0;
                __m256 _sum10 = _bias0;
                __m256 _sum11 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 4);
                __m256 _r02 = _mm256_loadu_ps(r0 + 8);
                __m256 _r03 = _mm256_loadu_ps(r0 + 12);
                __m256 _r04 = _mm256_loadu_ps(r0 + 16);
                __m256 _r05 = _mm256_loadu_ps(r0 + 20);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 4);
                __m256 _k02 = _mm256_loadu_ps(k0 + 8);
                __m256 _k03 = _mm256_loadu_ps(k0 + 12);
                __m256 _k04 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum00 = vmlaq_f32(_sum00, _k00, _r00);
                _sum00 = vmlaq_f32(_sum00, _k01, _r01);
                _sum00 = vmlaq_f32(_sum00, _k02, _r02);
                _sum00 = vmlaq_f32(_sum00, _k03, _r03);
                _sum00 = vmlaq_f32(_sum00, _k04, _r04);
                _sum01 = vmlaq_f32(_sum01, _k00, _r01);
                _sum01 = vmlaq_f32(_sum01, _k01, _r02);
                _sum01 = vmlaq_f32(_sum01, _k02, _r03);
                _sum01 = vmlaq_f32(_sum01, _k03, _r04);
                _sum01 = vmlaq_f32(_sum01, _k04, _r05);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 4);
                __m256 _r12 = _mm256_loadu_ps(r1 + 8);
                __m256 _r13 = _mm256_loadu_ps(r1 + 12);
                __m256 _r14 = _mm256_loadu_ps(r1 + 16);
                __m256 _r15 = _mm256_loadu_ps(r1 + 20);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 4);
                __m256 _k12 = _mm256_loadu_ps(k0 + 8);
                __m256 _k13 = _mm256_loadu_ps(k0 + 12);
                __m256 _k14 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum10 = vmlaq_f32(_sum10, _k00, _r10);
                _sum10 = vmlaq_f32(_sum10, _k01, _r11);
                _sum10 = vmlaq_f32(_sum10, _k02, _r12);
                _sum10 = vmlaq_f32(_sum10, _k03, _r13);
                _sum10 = vmlaq_f32(_sum10, _k04, _r14);
                _sum11 = vmlaq_f32(_sum11, _k00, _r11);
                _sum11 = vmlaq_f32(_sum11, _k01, _r12);
                _sum11 = vmlaq_f32(_sum11, _k02, _r13);
                _sum11 = vmlaq_f32(_sum11, _k03, _r14);
                _sum11 = vmlaq_f32(_sum11, _k04, _r15);

                _sum00 = vmlaq_f32(_sum00, _k10, _r10);
                _sum00 = vmlaq_f32(_sum00, _k11, _r11);
                _sum00 = vmlaq_f32(_sum00, _k12, _r12);
                _sum00 = vmlaq_f32(_sum00, _k13, _r13);
                _sum00 = vmlaq_f32(_sum00, _k14, _r14);
                _sum01 = vmlaq_f32(_sum01, _k10, _r11);
                _sum01 = vmlaq_f32(_sum01, _k11, _r12);
                _sum01 = vmlaq_f32(_sum01, _k12, _r13);
                _sum01 = vmlaq_f32(_sum01, _k13, _r14);
                _sum01 = vmlaq_f32(_sum01, _k14, _r15);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 4);
                __m256 _r22 = _mm256_loadu_ps(r2 + 8);
                __m256 _r23 = _mm256_loadu_ps(r2 + 12);
                __m256 _r24 = _mm256_loadu_ps(r2 + 16);
                __m256 _r25 = _mm256_loadu_ps(r2 + 20);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 4);
                __m256 _k22 = _mm256_loadu_ps(k0 + 8);
                __m256 _k23 = _mm256_loadu_ps(k0 + 12);
                __m256 _k24 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum10 = vmlaq_f32(_sum10, _k10, _r20);
                _sum10 = vmlaq_f32(_sum10, _k11, _r21);
                _sum10 = vmlaq_f32(_sum10, _k12, _r22);
                _sum10 = vmlaq_f32(_sum10, _k13, _r23);
                _sum10 = vmlaq_f32(_sum10, _k14, _r24);
                _sum11 = vmlaq_f32(_sum11, _k10, _r21);
                _sum11 = vmlaq_f32(_sum11, _k11, _r22);
                _sum11 = vmlaq_f32(_sum11, _k12, _r23);
                _sum11 = vmlaq_f32(_sum11, _k13, _r24);
                _sum11 = vmlaq_f32(_sum11, _k14, _r25);

                _sum00 = vmlaq_f32(_sum00, _k20, _r20);
                _sum00 = vmlaq_f32(_sum00, _k21, _r21);
                _sum00 = vmlaq_f32(_sum00, _k22, _r22);
                _sum00 = vmlaq_f32(_sum00, _k23, _r23);
                _sum00 = vmlaq_f32(_sum00, _k24, _r24);
                _sum01 = vmlaq_f32(_sum01, _k20, _r21);
                _sum01 = vmlaq_f32(_sum01, _k21, _r22);
                _sum01 = vmlaq_f32(_sum01, _k22, _r23);
                _sum01 = vmlaq_f32(_sum01, _k23, _r24);
                _sum01 = vmlaq_f32(_sum01, _k24, _r25);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 4);
                __m256 _r32 = _mm256_loadu_ps(r3 + 8);
                __m256 _r33 = _mm256_loadu_ps(r3 + 12);
                __m256 _r34 = _mm256_loadu_ps(r3 + 16);
                __m256 _r35 = _mm256_loadu_ps(r3 + 20);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 4);
                __m256 _k32 = _mm256_loadu_ps(k0 + 8);
                __m256 _k33 = _mm256_loadu_ps(k0 + 12);
                __m256 _k34 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum10 = vmlaq_f32(_sum10, _k20, _r30);
                _sum10 = vmlaq_f32(_sum10, _k21, _r31);
                _sum10 = vmlaq_f32(_sum10, _k22, _r32);
                _sum10 = vmlaq_f32(_sum10, _k23, _r33);
                _sum10 = vmlaq_f32(_sum10, _k24, _r34);
                _sum11 = vmlaq_f32(_sum11, _k20, _r31);
                _sum11 = vmlaq_f32(_sum11, _k21, _r32);
                _sum11 = vmlaq_f32(_sum11, _k22, _r33);
                _sum11 = vmlaq_f32(_sum11, _k23, _r34);
                _sum11 = vmlaq_f32(_sum11, _k24, _r35);

                _sum00 = vmlaq_f32(_sum00, _k30, _r30);
                _sum00 = vmlaq_f32(_sum00, _k31, _r31);
                _sum00 = vmlaq_f32(_sum00, _k32, _r32);
                _sum00 = vmlaq_f32(_sum00, _k33, _r33);
                _sum00 = vmlaq_f32(_sum00, _k34, _r34);
                _sum01 = vmlaq_f32(_sum01, _k30, _r31);
                _sum01 = vmlaq_f32(_sum01, _k31, _r32);
                _sum01 = vmlaq_f32(_sum01, _k32, _r33);
                _sum01 = vmlaq_f32(_sum01, _k33, _r34);
                _sum01 = vmlaq_f32(_sum01, _k34, _r35);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 4);
                __m256 _r42 = _mm256_loadu_ps(r4 + 8);
                __m256 _r43 = _mm256_loadu_ps(r4 + 12);
                __m256 _r44 = _mm256_loadu_ps(r4 + 16);
                __m256 _r45 = _mm256_loadu_ps(r4 + 20);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 4);
                __m256 _k42 = _mm256_loadu_ps(k0 + 8);
                __m256 _k43 = _mm256_loadu_ps(k0 + 12);
                __m256 _k44 = _mm256_loadu_ps(k0 + 16);
                k0 -= 80;

                _sum10 = vmlaq_f32(_sum10, _k30, _r40);
                _sum10 = vmlaq_f32(_sum10, _k31, _r41);
                _sum10 = vmlaq_f32(_sum10, _k32, _r42);
                _sum10 = vmlaq_f32(_sum10, _k33, _r43);
                _sum10 = vmlaq_f32(_sum10, _k34, _r44);
                _sum11 = vmlaq_f32(_sum11, _k30, _r41);
                _sum11 = vmlaq_f32(_sum11, _k31, _r42);
                _sum11 = vmlaq_f32(_sum11, _k32, _r43);
                _sum11 = vmlaq_f32(_sum11, _k33, _r44);
                _sum11 = vmlaq_f32(_sum11, _k34, _r45);

                _sum00 = vmlaq_f32(_sum00, _k40, _r40);
                _sum00 = vmlaq_f32(_sum00, _k41, _r41);
                _sum00 = vmlaq_f32(_sum00, _k42, _r42);
                _sum00 = vmlaq_f32(_sum00, _k43, _r43);
                _sum00 = vmlaq_f32(_sum00, _k44, _r44);
                _sum01 = vmlaq_f32(_sum01, _k40, _r41);
                _sum01 = vmlaq_f32(_sum01, _k41, _r42);
                _sum01 = vmlaq_f32(_sum01, _k42, _r43);
                _sum01 = vmlaq_f32(_sum01, _k43, _r44);
                _sum01 = vmlaq_f32(_sum01, _k44, _r45);

                __m256 _r50 = _mm256_loadu_ps(r5);
                __m256 _r51 = _mm256_loadu_ps(r5 + 4);
                __m256 _r52 = _mm256_loadu_ps(r5 + 8);
                __m256 _r53 = _mm256_loadu_ps(r5 + 12);
                __m256 _r54 = _mm256_loadu_ps(r5 + 16);
                __m256 _r55 = _mm256_loadu_ps(r5 + 20);

                _sum10 = vmlaq_f32(_sum10, _k40, _r50);
                _sum10 = vmlaq_f32(_sum10, _k41, _r51);
                _sum10 = vmlaq_f32(_sum10, _k42, _r52);
                _sum10 = vmlaq_f32(_sum10, _k43, _r53);
                _sum10 = vmlaq_f32(_sum10, _k44, _r54);
                _sum11 = vmlaq_f32(_sum11, _k40, _r51);
                _sum11 = vmlaq_f32(_sum11, _k41, _r52);
                _sum11 = vmlaq_f32(_sum11, _k42, _r53);
                _sum11 = vmlaq_f32(_sum11, _k43, _r54);
                _sum11 = vmlaq_f32(_sum11, _k44, _r55);

                _mm256_storeu_ps(outptr0, _sum00);
                _mm256_storeu_ps(outptr0 + 4, _sum01);
                _mm256_storeu_ps(outptr1, _sum10);
                _mm256_storeu_ps(outptr1 + 4, _sum11);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                r5 += 8;
                outptr0 += 8;
                outptr1 += 8;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 4);
                __m256 _r02 = _mm256_loadu_ps(r0 + 8);
                __m256 _r03 = _mm256_loadu_ps(r0 + 12);
                __m256 _r04 = _mm256_loadu_ps(r0 + 16);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 4);
                __m256 _k02 = _mm256_loadu_ps(k0 + 8);
                __m256 _k03 = _mm256_loadu_ps(k0 + 12);
                __m256 _k04 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 4);
                __m256 _r12 = _mm256_loadu_ps(r1 + 8);
                __m256 _r13 = _mm256_loadu_ps(r1 + 12);
                __m256 _r14 = _mm256_loadu_ps(r1 + 16);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 4);
                __m256 _k12 = _mm256_loadu_ps(k0 + 8);
                __m256 _k13 = _mm256_loadu_ps(k0 + 12);
                __m256 _k14 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum1 = vmlaq_f32(_sum1, _k00, _r10);
                _sum1 = vmlaq_f32(_sum1, _k01, _r11);
                _sum1 = vmlaq_f32(_sum1, _k02, _r12);
                _sum1 = vmlaq_f32(_sum1, _k03, _r13);
                _sum1 = vmlaq_f32(_sum1, _k04, _r14);

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 4);
                __m256 _r22 = _mm256_loadu_ps(r2 + 8);
                __m256 _r23 = _mm256_loadu_ps(r2 + 12);
                __m256 _r24 = _mm256_loadu_ps(r2 + 16);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 4);
                __m256 _k22 = _mm256_loadu_ps(k0 + 8);
                __m256 _k23 = _mm256_loadu_ps(k0 + 12);
                __m256 _k24 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum1 = vmlaq_f32(_sum1, _k10, _r20);
                _sum1 = vmlaq_f32(_sum1, _k11, _r21);
                _sum1 = vmlaq_f32(_sum1, _k12, _r22);
                _sum1 = vmlaq_f32(_sum1, _k13, _r23);
                _sum1 = vmlaq_f32(_sum1, _k14, _r24);

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 4);
                __m256 _r32 = _mm256_loadu_ps(r3 + 8);
                __m256 _r33 = _mm256_loadu_ps(r3 + 12);
                __m256 _r34 = _mm256_loadu_ps(r3 + 16);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 4);
                __m256 _k32 = _mm256_loadu_ps(k0 + 8);
                __m256 _k33 = _mm256_loadu_ps(k0 + 12);
                __m256 _k34 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum1 = vmlaq_f32(_sum1, _k20, _r30);
                _sum1 = vmlaq_f32(_sum1, _k21, _r31);
                _sum1 = vmlaq_f32(_sum1, _k22, _r32);
                _sum1 = vmlaq_f32(_sum1, _k23, _r33);
                _sum1 = vmlaq_f32(_sum1, _k24, _r34);

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 4);
                __m256 _r42 = _mm256_loadu_ps(r4 + 8);
                __m256 _r43 = _mm256_loadu_ps(r4 + 12);
                __m256 _r44 = _mm256_loadu_ps(r4 + 16);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 4);
                __m256 _k42 = _mm256_loadu_ps(k0 + 8);
                __m256 _k43 = _mm256_loadu_ps(k0 + 12);
                __m256 _k44 = _mm256_loadu_ps(k0 + 16);
                k0 -= 80;

                _sum1 = vmlaq_f32(_sum1, _k30, _r40);
                _sum1 = vmlaq_f32(_sum1, _k31, _r41);
                _sum1 = vmlaq_f32(_sum1, _k32, _r42);
                _sum1 = vmlaq_f32(_sum1, _k33, _r43);
                _sum1 = vmlaq_f32(_sum1, _k34, _r44);

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);

                __m256 _r50 = _mm256_loadu_ps(r5);
                __m256 _r51 = _mm256_loadu_ps(r5 + 4);
                __m256 _r52 = _mm256_loadu_ps(r5 + 8);
                __m256 _r53 = _mm256_loadu_ps(r5 + 12);
                __m256 _r54 = _mm256_loadu_ps(r5 + 16);

                _sum1 = vmlaq_f32(_sum1, _k40, _r50);
                _sum1 = vmlaq_f32(_sum1, _k41, _r51);
                _sum1 = vmlaq_f32(_sum1, _k42, _r52);
                _sum1 = vmlaq_f32(_sum1, _k43, _r53);
                _sum1 = vmlaq_f32(_sum1, _k44, _r54);

                _mm256_storeu_ps(outptr0, _sum0);
                _mm256_storeu_ps(outptr1, _sum1);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
                outptr0 += 4;
                outptr1 += 4;
            }

            r0 += 4 * 4 + w * 4;
            r1 += 4 * 4 + w * 4;
            r2 += 4 * 4 + w * 4;
            r3 += 4 * 4 + w * 4;
            r4 += 4 * 4 + w * 4;
            r5 += 4 * 4 + w * 4;

            outptr0 += outw * 4;
            outptr1 += outw * 4;
        }
#endif // __aarch64__
        for (; i < outh; i++)
        {
            int j = 0;

            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;
                __m256 _sum2 = _bias0;
                __m256 _sum3 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 4);
                __m256 _r02 = _mm256_loadu_ps(r0 + 8);
                __m256 _r03 = _mm256_loadu_ps(r0 + 12);
                __m256 _r04 = _mm256_loadu_ps(r0 + 16);
                __m256 _r05 = _mm256_loadu_ps(r0 + 20);
                __m256 _r06 = _mm256_loadu_ps(r0 + 24);
                __m256 _r07 = _mm256_loadu_ps(r0 + 28);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 4);
                __m256 _k02 = _mm256_loadu_ps(k0 + 8);
                __m256 _k03 = _mm256_loadu_ps(k0 + 12);
                __m256 _k04 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);
                _sum1 = vmlaq_f32(_sum1, _k00, _r01);
                _sum1 = vmlaq_f32(_sum1, _k01, _r02);
                _sum1 = vmlaq_f32(_sum1, _k02, _r03);
                _sum1 = vmlaq_f32(_sum1, _k03, _r04);
                _sum1 = vmlaq_f32(_sum1, _k04, _r05);
                _sum2 = vmlaq_f32(_sum2, _k00, _r02);
                _sum2 = vmlaq_f32(_sum2, _k01, _r03);
                _sum2 = vmlaq_f32(_sum2, _k02, _r04);
                _sum2 = vmlaq_f32(_sum2, _k03, _r05);
                _sum2 = vmlaq_f32(_sum2, _k04, _r06);
                _sum3 = vmlaq_f32(_sum3, _k00, _r03);
                _sum3 = vmlaq_f32(_sum3, _k01, _r04);
                _sum3 = vmlaq_f32(_sum3, _k02, _r05);
                _sum3 = vmlaq_f32(_sum3, _k03, _r06);
                _sum3 = vmlaq_f32(_sum3, _k04, _r07);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 4);
                __m256 _r12 = _mm256_loadu_ps(r1 + 8);
                __m256 _r13 = _mm256_loadu_ps(r1 + 12);
                __m256 _r14 = _mm256_loadu_ps(r1 + 16);
                __m256 _r15 = _mm256_loadu_ps(r1 + 20);
                __m256 _r16 = _mm256_loadu_ps(r1 + 24);
                __m256 _r17 = _mm256_loadu_ps(r1 + 28);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 4);
                __m256 _k12 = _mm256_loadu_ps(k0 + 8);
                __m256 _k13 = _mm256_loadu_ps(k0 + 12);
                __m256 _k14 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);
                _sum1 = vmlaq_f32(_sum1, _k10, _r11);
                _sum1 = vmlaq_f32(_sum1, _k11, _r12);
                _sum1 = vmlaq_f32(_sum1, _k12, _r13);
                _sum1 = vmlaq_f32(_sum1, _k13, _r14);
                _sum1 = vmlaq_f32(_sum1, _k14, _r15);
                _sum2 = vmlaq_f32(_sum2, _k10, _r12);
                _sum2 = vmlaq_f32(_sum2, _k11, _r13);
                _sum2 = vmlaq_f32(_sum2, _k12, _r14);
                _sum2 = vmlaq_f32(_sum2, _k13, _r15);
                _sum2 = vmlaq_f32(_sum2, _k14, _r16);
                _sum3 = vmlaq_f32(_sum3, _k10, _r13);
                _sum3 = vmlaq_f32(_sum3, _k11, _r14);
                _sum3 = vmlaq_f32(_sum3, _k12, _r15);
                _sum3 = vmlaq_f32(_sum3, _k13, _r16);
                _sum3 = vmlaq_f32(_sum3, _k14, _r17);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 4);
                __m256 _r22 = _mm256_loadu_ps(r2 + 8);
                __m256 _r23 = _mm256_loadu_ps(r2 + 12);
                __m256 _r24 = _mm256_loadu_ps(r2 + 16);
                __m256 _r25 = _mm256_loadu_ps(r2 + 20);
                __m256 _r26 = _mm256_loadu_ps(r2 + 24);
                __m256 _r27 = _mm256_loadu_ps(r2 + 28);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 4);
                __m256 _k22 = _mm256_loadu_ps(k0 + 8);
                __m256 _k23 = _mm256_loadu_ps(k0 + 12);
                __m256 _k24 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);
                _sum1 = vmlaq_f32(_sum1, _k20, _r21);
                _sum1 = vmlaq_f32(_sum1, _k21, _r22);
                _sum1 = vmlaq_f32(_sum1, _k22, _r23);
                _sum1 = vmlaq_f32(_sum1, _k23, _r24);
                _sum1 = vmlaq_f32(_sum1, _k24, _r25);
                _sum2 = vmlaq_f32(_sum2, _k20, _r22);
                _sum2 = vmlaq_f32(_sum2, _k21, _r23);
                _sum2 = vmlaq_f32(_sum2, _k22, _r24);
                _sum2 = vmlaq_f32(_sum2, _k23, _r25);
                _sum2 = vmlaq_f32(_sum2, _k24, _r26);
                _sum3 = vmlaq_f32(_sum3, _k20, _r23);
                _sum3 = vmlaq_f32(_sum3, _k21, _r24);
                _sum3 = vmlaq_f32(_sum3, _k22, _r25);
                _sum3 = vmlaq_f32(_sum3, _k23, _r26);
                _sum3 = vmlaq_f32(_sum3, _k24, _r27);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 4);
                __m256 _r32 = _mm256_loadu_ps(r3 + 8);
                __m256 _r33 = _mm256_loadu_ps(r3 + 12);
                __m256 _r34 = _mm256_loadu_ps(r3 + 16);
                __m256 _r35 = _mm256_loadu_ps(r3 + 20);
                __m256 _r36 = _mm256_loadu_ps(r3 + 24);
                __m256 _r37 = _mm256_loadu_ps(r3 + 28);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 4);
                __m256 _k32 = _mm256_loadu_ps(k0 + 8);
                __m256 _k33 = _mm256_loadu_ps(k0 + 12);
                __m256 _k34 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);
                _sum1 = vmlaq_f32(_sum1, _k30, _r31);
                _sum1 = vmlaq_f32(_sum1, _k31, _r32);
                _sum1 = vmlaq_f32(_sum1, _k32, _r33);
                _sum1 = vmlaq_f32(_sum1, _k33, _r34);
                _sum1 = vmlaq_f32(_sum1, _k34, _r35);
                _sum2 = vmlaq_f32(_sum2, _k30, _r32);
                _sum2 = vmlaq_f32(_sum2, _k31, _r33);
                _sum2 = vmlaq_f32(_sum2, _k32, _r34);
                _sum2 = vmlaq_f32(_sum2, _k33, _r35);
                _sum2 = vmlaq_f32(_sum2, _k34, _r36);
                _sum3 = vmlaq_f32(_sum3, _k30, _r33);
                _sum3 = vmlaq_f32(_sum3, _k31, _r34);
                _sum3 = vmlaq_f32(_sum3, _k32, _r35);
                _sum3 = vmlaq_f32(_sum3, _k33, _r36);
                _sum3 = vmlaq_f32(_sum3, _k34, _r37);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 4);
                __m256 _r42 = _mm256_loadu_ps(r4 + 8);
                __m256 _r43 = _mm256_loadu_ps(r4 + 12);
                __m256 _r44 = _mm256_loadu_ps(r4 + 16);
                __m256 _r45 = _mm256_loadu_ps(r4 + 20);
                __m256 _r46 = _mm256_loadu_ps(r4 + 24);
                __m256 _r47 = _mm256_loadu_ps(r4 + 28);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 4);
                __m256 _k42 = _mm256_loadu_ps(k0 + 8);
                __m256 _k43 = _mm256_loadu_ps(k0 + 12);
                __m256 _k44 = _mm256_loadu_ps(k0 + 16);
                k0 -= 80;

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);
                _sum1 = vmlaq_f32(_sum1, _k40, _r41);
                _sum1 = vmlaq_f32(_sum1, _k41, _r42);
                _sum1 = vmlaq_f32(_sum1, _k42, _r43);
                _sum1 = vmlaq_f32(_sum1, _k43, _r44);
                _sum1 = vmlaq_f32(_sum1, _k44, _r45);
                _sum2 = vmlaq_f32(_sum2, _k40, _r42);
                _sum2 = vmlaq_f32(_sum2, _k41, _r43);
                _sum2 = vmlaq_f32(_sum2, _k42, _r44);
                _sum2 = vmlaq_f32(_sum2, _k43, _r45);
                _sum2 = vmlaq_f32(_sum2, _k44, _r46);
                _sum3 = vmlaq_f32(_sum3, _k40, _r43);
                _sum3 = vmlaq_f32(_sum3, _k41, _r44);
                _sum3 = vmlaq_f32(_sum3, _k42, _r45);
                _sum3 = vmlaq_f32(_sum3, _k43, _r46);
                _sum3 = vmlaq_f32(_sum3, _k44, _r47);

                _mm256_storeu_ps(outptr0, _sum0);
                _mm256_storeu_ps(outptr0 + 4, _sum1);
                _mm256_storeu_ps(outptr0 + 8, _sum2);
                _mm256_storeu_ps(outptr0 + 12, _sum3);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                r4 += 16;
                outptr0 += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 4);
                __m256 _r02 = _mm256_loadu_ps(r0 + 8);
                __m256 _r03 = _mm256_loadu_ps(r0 + 12);
                __m256 _r04 = _mm256_loadu_ps(r0 + 16);
                __m256 _r05 = _mm256_loadu_ps(r0 + 20);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 4);
                __m256 _k02 = _mm256_loadu_ps(k0 + 8);
                __m256 _k03 = _mm256_loadu_ps(k0 + 12);
                __m256 _k04 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);
                _sum1 = vmlaq_f32(_sum1, _k00, _r01);
                _sum1 = vmlaq_f32(_sum1, _k01, _r02);
                _sum1 = vmlaq_f32(_sum1, _k02, _r03);
                _sum1 = vmlaq_f32(_sum1, _k03, _r04);
                _sum1 = vmlaq_f32(_sum1, _k04, _r05);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 4);
                __m256 _r12 = _mm256_loadu_ps(r1 + 8);
                __m256 _r13 = _mm256_loadu_ps(r1 + 12);
                __m256 _r14 = _mm256_loadu_ps(r1 + 16);
                __m256 _r15 = _mm256_loadu_ps(r1 + 20);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 4);
                __m256 _k12 = _mm256_loadu_ps(k0 + 8);
                __m256 _k13 = _mm256_loadu_ps(k0 + 12);
                __m256 _k14 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);
                _sum1 = vmlaq_f32(_sum1, _k10, _r11);
                _sum1 = vmlaq_f32(_sum1, _k11, _r12);
                _sum1 = vmlaq_f32(_sum1, _k12, _r13);
                _sum1 = vmlaq_f32(_sum1, _k13, _r14);
                _sum1 = vmlaq_f32(_sum1, _k14, _r15);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 4);
                __m256 _r22 = _mm256_loadu_ps(r2 + 8);
                __m256 _r23 = _mm256_loadu_ps(r2 + 12);
                __m256 _r24 = _mm256_loadu_ps(r2 + 16);
                __m256 _r25 = _mm256_loadu_ps(r2 + 20);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 4);
                __m256 _k22 = _mm256_loadu_ps(k0 + 8);
                __m256 _k23 = _mm256_loadu_ps(k0 + 12);
                __m256 _k24 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);
                _sum1 = vmlaq_f32(_sum1, _k20, _r21);
                _sum1 = vmlaq_f32(_sum1, _k21, _r22);
                _sum1 = vmlaq_f32(_sum1, _k22, _r23);
                _sum1 = vmlaq_f32(_sum1, _k23, _r24);
                _sum1 = vmlaq_f32(_sum1, _k24, _r25);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 4);
                __m256 _r32 = _mm256_loadu_ps(r3 + 8);
                __m256 _r33 = _mm256_loadu_ps(r3 + 12);
                __m256 _r34 = _mm256_loadu_ps(r3 + 16);
                __m256 _r35 = _mm256_loadu_ps(r3 + 20);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 4);
                __m256 _k32 = _mm256_loadu_ps(k0 + 8);
                __m256 _k33 = _mm256_loadu_ps(k0 + 12);
                __m256 _k34 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);
                _sum1 = vmlaq_f32(_sum1, _k30, _r31);
                _sum1 = vmlaq_f32(_sum1, _k31, _r32);
                _sum1 = vmlaq_f32(_sum1, _k32, _r33);
                _sum1 = vmlaq_f32(_sum1, _k33, _r34);
                _sum1 = vmlaq_f32(_sum1, _k34, _r35);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 4);
                __m256 _r42 = _mm256_loadu_ps(r4 + 8);
                __m256 _r43 = _mm256_loadu_ps(r4 + 12);
                __m256 _r44 = _mm256_loadu_ps(r4 + 16);
                __m256 _r45 = _mm256_loadu_ps(r4 + 20);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 4);
                __m256 _k42 = _mm256_loadu_ps(k0 + 8);
                __m256 _k43 = _mm256_loadu_ps(k0 + 12);
                __m256 _k44 = _mm256_loadu_ps(k0 + 16);
                k0 -= 80;

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);
                _sum1 = vmlaq_f32(_sum1, _k40, _r41);
                _sum1 = vmlaq_f32(_sum1, _k41, _r42);
                _sum1 = vmlaq_f32(_sum1, _k42, _r43);
                _sum1 = vmlaq_f32(_sum1, _k43, _r44);
                _sum1 = vmlaq_f32(_sum1, _k44, _r45);

                _mm256_storeu_ps(outptr0, _sum0);
                _mm256_storeu_ps(outptr0 + 4, _sum1);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                outptr0 += 8;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 4);
                __m256 _r02 = _mm256_loadu_ps(r0 + 8);
                __m256 _r03 = _mm256_loadu_ps(r0 + 12);
                __m256 _r04 = _mm256_loadu_ps(r0 + 16);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 4);
                __m256 _k02 = _mm256_loadu_ps(k0 + 8);
                __m256 _k03 = _mm256_loadu_ps(k0 + 12);
                __m256 _k04 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 4);
                __m256 _r12 = _mm256_loadu_ps(r1 + 8);
                __m256 _r13 = _mm256_loadu_ps(r1 + 12);
                __m256 _r14 = _mm256_loadu_ps(r1 + 16);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 4);
                __m256 _k12 = _mm256_loadu_ps(k0 + 8);
                __m256 _k13 = _mm256_loadu_ps(k0 + 12);
                __m256 _k14 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 4);
                __m256 _r22 = _mm256_loadu_ps(r2 + 8);
                __m256 _r23 = _mm256_loadu_ps(r2 + 12);
                __m256 _r24 = _mm256_loadu_ps(r2 + 16);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 4);
                __m256 _k22 = _mm256_loadu_ps(k0 + 8);
                __m256 _k23 = _mm256_loadu_ps(k0 + 12);
                __m256 _k24 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 4);
                __m256 _r32 = _mm256_loadu_ps(r3 + 8);
                __m256 _r33 = _mm256_loadu_ps(r3 + 12);
                __m256 _r34 = _mm256_loadu_ps(r3 + 16);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 4);
                __m256 _k32 = _mm256_loadu_ps(k0 + 8);
                __m256 _k33 = _mm256_loadu_ps(k0 + 12);
                __m256 _k34 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 4);
                __m256 _r42 = _mm256_loadu_ps(r4 + 8);
                __m256 _r43 = _mm256_loadu_ps(r4 + 12);
                __m256 _r44 = _mm256_loadu_ps(r4 + 16);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 4);
                __m256 _k42 = _mm256_loadu_ps(k0 + 8);
                __m256 _k43 = _mm256_loadu_ps(k0 + 12);
                __m256 _k44 = _mm256_loadu_ps(k0 + 16);
                k0 -= 80;

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);

                _mm256_storeu_ps(outptr0, _sum0);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                outptr0 += 4;
            }

            r0 += 4 * 4;
            r1 += 4 * 4;
            r2 += 4 * 4;
            r3 += 4 * 4;
            r4 += 4 * 4;
        }
    }
}

static void convdw5x5s2_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + g * 4) : _mm256_set1_ps(0.f);

        const float* k0 = kernel.row(g);

        float* outptr0 = out;

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        const float* r3 = img0.row(3);
        const float* r4 = img0.row(4);

        int i = 0;

        for (; i < outh; i++)
        {
            int j = 0;

            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;
                __m256 _sum2 = _bias0;
                __m256 _sum3 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 4);
                __m256 _r02 = _mm256_loadu_ps(r0 + 8);
                __m256 _r03 = _mm256_loadu_ps(r0 + 12);
                __m256 _r04 = _mm256_loadu_ps(r0 + 16);
                __m256 _r05 = _mm256_loadu_ps(r0 + 20);
                __m256 _r06 = _mm256_loadu_ps(r0 + 24);
                __m256 _r07 = _mm256_loadu_ps(r0 + 28);
                __m256 _r08 = _mm256_loadu_ps(r0 + 32);
                __m256 _r09 = _mm256_loadu_ps(r0 + 36);
                __m256 _r010 = _mm256_loadu_ps(r0 + 40);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 4);
                __m256 _k02 = _mm256_loadu_ps(k0 + 8);
                __m256 _k03 = _mm256_loadu_ps(k0 + 12);
                __m256 _k04 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);
                _sum1 = vmlaq_f32(_sum1, _k00, _r02);
                _sum1 = vmlaq_f32(_sum1, _k01, _r03);
                _sum1 = vmlaq_f32(_sum1, _k02, _r04);
                _sum1 = vmlaq_f32(_sum1, _k03, _r05);
                _sum1 = vmlaq_f32(_sum1, _k04, _r06);
                _sum2 = vmlaq_f32(_sum2, _k00, _r04);
                _sum2 = vmlaq_f32(_sum2, _k01, _r05);
                _sum2 = vmlaq_f32(_sum2, _k02, _r06);
                _sum2 = vmlaq_f32(_sum2, _k03, _r07);
                _sum2 = vmlaq_f32(_sum2, _k04, _r08);
                _sum3 = vmlaq_f32(_sum3, _k00, _r06);
                _sum3 = vmlaq_f32(_sum3, _k01, _r07);
                _sum3 = vmlaq_f32(_sum3, _k02, _r08);
                _sum3 = vmlaq_f32(_sum3, _k03, _r09);
                _sum3 = vmlaq_f32(_sum3, _k04, _r010);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 4);
                __m256 _r12 = _mm256_loadu_ps(r1 + 8);
                __m256 _r13 = _mm256_loadu_ps(r1 + 12);
                __m256 _r14 = _mm256_loadu_ps(r1 + 16);
                __m256 _r15 = _mm256_loadu_ps(r1 + 20);
                __m256 _r16 = _mm256_loadu_ps(r1 + 24);
                __m256 _r17 = _mm256_loadu_ps(r1 + 28);
                __m256 _r18 = _mm256_loadu_ps(r1 + 32);
                __m256 _r19 = _mm256_loadu_ps(r1 + 36);
                __m256 _r110 = _mm256_loadu_ps(r1 + 40);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 4);
                __m256 _k12 = _mm256_loadu_ps(k0 + 8);
                __m256 _k13 = _mm256_loadu_ps(k0 + 12);
                __m256 _k14 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);
                _sum1 = vmlaq_f32(_sum1, _k10, _r12);
                _sum1 = vmlaq_f32(_sum1, _k11, _r13);
                _sum1 = vmlaq_f32(_sum1, _k12, _r14);
                _sum1 = vmlaq_f32(_sum1, _k13, _r15);
                _sum1 = vmlaq_f32(_sum1, _k14, _r16);
                _sum2 = vmlaq_f32(_sum2, _k10, _r14);
                _sum2 = vmlaq_f32(_sum2, _k11, _r15);
                _sum2 = vmlaq_f32(_sum2, _k12, _r16);
                _sum2 = vmlaq_f32(_sum2, _k13, _r17);
                _sum2 = vmlaq_f32(_sum2, _k14, _r18);
                _sum3 = vmlaq_f32(_sum3, _k10, _r16);
                _sum3 = vmlaq_f32(_sum3, _k11, _r17);
                _sum3 = vmlaq_f32(_sum3, _k12, _r18);
                _sum3 = vmlaq_f32(_sum3, _k13, _r19);
                _sum3 = vmlaq_f32(_sum3, _k14, _r110);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 4);
                __m256 _r22 = _mm256_loadu_ps(r2 + 8);
                __m256 _r23 = _mm256_loadu_ps(r2 + 12);
                __m256 _r24 = _mm256_loadu_ps(r2 + 16);
                __m256 _r25 = _mm256_loadu_ps(r2 + 20);
                __m256 _r26 = _mm256_loadu_ps(r2 + 24);
                __m256 _r27 = _mm256_loadu_ps(r2 + 28);
                __m256 _r28 = _mm256_loadu_ps(r2 + 32);
                __m256 _r29 = _mm256_loadu_ps(r2 + 36);
                __m256 _r210 = _mm256_loadu_ps(r2 + 40);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 4);
                __m256 _k22 = _mm256_loadu_ps(k0 + 8);
                __m256 _k23 = _mm256_loadu_ps(k0 + 12);
                __m256 _k24 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);
                _sum1 = vmlaq_f32(_sum1, _k20, _r22);
                _sum1 = vmlaq_f32(_sum1, _k21, _r23);
                _sum1 = vmlaq_f32(_sum1, _k22, _r24);
                _sum1 = vmlaq_f32(_sum1, _k23, _r25);
                _sum1 = vmlaq_f32(_sum1, _k24, _r26);
                _sum2 = vmlaq_f32(_sum2, _k20, _r24);
                _sum2 = vmlaq_f32(_sum2, _k21, _r25);
                _sum2 = vmlaq_f32(_sum2, _k22, _r26);
                _sum2 = vmlaq_f32(_sum2, _k23, _r27);
                _sum2 = vmlaq_f32(_sum2, _k24, _r28);
                _sum3 = vmlaq_f32(_sum3, _k20, _r26);
                _sum3 = vmlaq_f32(_sum3, _k21, _r27);
                _sum3 = vmlaq_f32(_sum3, _k22, _r28);
                _sum3 = vmlaq_f32(_sum3, _k23, _r29);
                _sum3 = vmlaq_f32(_sum3, _k24, _r210);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 4);
                __m256 _r32 = _mm256_loadu_ps(r3 + 8);
                __m256 _r33 = _mm256_loadu_ps(r3 + 12);
                __m256 _r34 = _mm256_loadu_ps(r3 + 16);
                __m256 _r35 = _mm256_loadu_ps(r3 + 20);
                __m256 _r36 = _mm256_loadu_ps(r3 + 24);
                __m256 _r37 = _mm256_loadu_ps(r3 + 28);
                __m256 _r38 = _mm256_loadu_ps(r3 + 32);
                __m256 _r39 = _mm256_loadu_ps(r3 + 36);
                __m256 _r310 = _mm256_loadu_ps(r3 + 40);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 4);
                __m256 _k32 = _mm256_loadu_ps(k0 + 8);
                __m256 _k33 = _mm256_loadu_ps(k0 + 12);
                __m256 _k34 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);
                _sum1 = vmlaq_f32(_sum1, _k30, _r32);
                _sum1 = vmlaq_f32(_sum1, _k31, _r33);
                _sum1 = vmlaq_f32(_sum1, _k32, _r34);
                _sum1 = vmlaq_f32(_sum1, _k33, _r35);
                _sum1 = vmlaq_f32(_sum1, _k34, _r36);
                _sum2 = vmlaq_f32(_sum2, _k30, _r34);
                _sum2 = vmlaq_f32(_sum2, _k31, _r35);
                _sum2 = vmlaq_f32(_sum2, _k32, _r36);
                _sum2 = vmlaq_f32(_sum2, _k33, _r37);
                _sum2 = vmlaq_f32(_sum2, _k34, _r38);
                _sum3 = vmlaq_f32(_sum3, _k30, _r36);
                _sum3 = vmlaq_f32(_sum3, _k31, _r37);
                _sum3 = vmlaq_f32(_sum3, _k32, _r38);
                _sum3 = vmlaq_f32(_sum3, _k33, _r39);
                _sum3 = vmlaq_f32(_sum3, _k34, _r310);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 4);
                __m256 _r42 = _mm256_loadu_ps(r4 + 8);
                __m256 _r43 = _mm256_loadu_ps(r4 + 12);
                __m256 _r44 = _mm256_loadu_ps(r4 + 16);
                __m256 _r45 = _mm256_loadu_ps(r4 + 20);
                __m256 _r46 = _mm256_loadu_ps(r4 + 24);
                __m256 _r47 = _mm256_loadu_ps(r4 + 28);
                __m256 _r48 = _mm256_loadu_ps(r4 + 32);
                __m256 _r49 = _mm256_loadu_ps(r4 + 36);
                __m256 _r410 = _mm256_loadu_ps(r4 + 40);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 4);
                __m256 _k42 = _mm256_loadu_ps(k0 + 8);
                __m256 _k43 = _mm256_loadu_ps(k0 + 12);
                __m256 _k44 = _mm256_loadu_ps(k0 + 16);
                k0 -= 80;

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);
                _sum1 = vmlaq_f32(_sum1, _k40, _r42);
                _sum1 = vmlaq_f32(_sum1, _k41, _r43);
                _sum1 = vmlaq_f32(_sum1, _k42, _r44);
                _sum1 = vmlaq_f32(_sum1, _k43, _r45);
                _sum1 = vmlaq_f32(_sum1, _k44, _r46);
                _sum2 = vmlaq_f32(_sum2, _k40, _r44);
                _sum2 = vmlaq_f32(_sum2, _k41, _r45);
                _sum2 = vmlaq_f32(_sum2, _k42, _r46);
                _sum2 = vmlaq_f32(_sum2, _k43, _r47);
                _sum2 = vmlaq_f32(_sum2, _k44, _r48);
                _sum3 = vmlaq_f32(_sum3, _k40, _r46);
                _sum3 = vmlaq_f32(_sum3, _k41, _r47);
                _sum3 = vmlaq_f32(_sum3, _k42, _r48);
                _sum3 = vmlaq_f32(_sum3, _k43, _r49);
                _sum3 = vmlaq_f32(_sum3, _k44, _r410);

                _mm256_storeu_ps(outptr0, _sum0);
                _mm256_storeu_ps(outptr0 + 4, _sum1);
                _mm256_storeu_ps(outptr0 + 8, _sum2);
                _mm256_storeu_ps(outptr0 + 12, _sum3);

                r0 += 8 * 4;
                r1 += 8 * 4;
                r2 += 8 * 4;
                r3 += 8 * 4;
                r4 += 8 * 4;
                outptr0 += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 4);
                __m256 _r02 = _mm256_loadu_ps(r0 + 8);
                __m256 _r03 = _mm256_loadu_ps(r0 + 12);
                __m256 _r04 = _mm256_loadu_ps(r0 + 16);
                __m256 _r05 = _mm256_loadu_ps(r0 + 20);
                __m256 _r06 = _mm256_loadu_ps(r0 + 24);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 4);
                __m256 _k02 = _mm256_loadu_ps(k0 + 8);
                __m256 _k03 = _mm256_loadu_ps(k0 + 12);
                __m256 _k04 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);
                _sum1 = vmlaq_f32(_sum1, _k00, _r02);
                _sum1 = vmlaq_f32(_sum1, _k01, _r03);
                _sum1 = vmlaq_f32(_sum1, _k02, _r04);
                _sum1 = vmlaq_f32(_sum1, _k03, _r05);
                _sum1 = vmlaq_f32(_sum1, _k04, _r06);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 4);
                __m256 _r12 = _mm256_loadu_ps(r1 + 8);
                __m256 _r13 = _mm256_loadu_ps(r1 + 12);
                __m256 _r14 = _mm256_loadu_ps(r1 + 16);
                __m256 _r15 = _mm256_loadu_ps(r1 + 20);
                __m256 _r16 = _mm256_loadu_ps(r1 + 24);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 4);
                __m256 _k12 = _mm256_loadu_ps(k0 + 8);
                __m256 _k13 = _mm256_loadu_ps(k0 + 12);
                __m256 _k14 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);
                _sum1 = vmlaq_f32(_sum1, _k10, _r12);
                _sum1 = vmlaq_f32(_sum1, _k11, _r13);
                _sum1 = vmlaq_f32(_sum1, _k12, _r14);
                _sum1 = vmlaq_f32(_sum1, _k13, _r15);
                _sum1 = vmlaq_f32(_sum1, _k14, _r16);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 4);
                __m256 _r22 = _mm256_loadu_ps(r2 + 8);
                __m256 _r23 = _mm256_loadu_ps(r2 + 12);
                __m256 _r24 = _mm256_loadu_ps(r2 + 16);
                __m256 _r25 = _mm256_loadu_ps(r2 + 20);
                __m256 _r26 = _mm256_loadu_ps(r2 + 24);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 4);
                __m256 _k22 = _mm256_loadu_ps(k0 + 8);
                __m256 _k23 = _mm256_loadu_ps(k0 + 12);
                __m256 _k24 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);
                _sum1 = vmlaq_f32(_sum1, _k20, _r22);
                _sum1 = vmlaq_f32(_sum1, _k21, _r23);
                _sum1 = vmlaq_f32(_sum1, _k22, _r24);
                _sum1 = vmlaq_f32(_sum1, _k23, _r25);
                _sum1 = vmlaq_f32(_sum1, _k24, _r26);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 4);
                __m256 _r32 = _mm256_loadu_ps(r3 + 8);
                __m256 _r33 = _mm256_loadu_ps(r3 + 12);
                __m256 _r34 = _mm256_loadu_ps(r3 + 16);
                __m256 _r35 = _mm256_loadu_ps(r3 + 20);
                __m256 _r36 = _mm256_loadu_ps(r3 + 24);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 4);
                __m256 _k32 = _mm256_loadu_ps(k0 + 8);
                __m256 _k33 = _mm256_loadu_ps(k0 + 12);
                __m256 _k34 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);
                _sum1 = vmlaq_f32(_sum1, _k30, _r32);
                _sum1 = vmlaq_f32(_sum1, _k31, _r33);
                _sum1 = vmlaq_f32(_sum1, _k32, _r34);
                _sum1 = vmlaq_f32(_sum1, _k33, _r35);
                _sum1 = vmlaq_f32(_sum1, _k34, _r36);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 4);
                __m256 _r42 = _mm256_loadu_ps(r4 + 8);
                __m256 _r43 = _mm256_loadu_ps(r4 + 12);
                __m256 _r44 = _mm256_loadu_ps(r4 + 16);
                __m256 _r45 = _mm256_loadu_ps(r4 + 20);
                __m256 _r46 = _mm256_loadu_ps(r4 + 24);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 4);
                __m256 _k42 = _mm256_loadu_ps(k0 + 8);
                __m256 _k43 = _mm256_loadu_ps(k0 + 12);
                __m256 _k44 = _mm256_loadu_ps(k0 + 16);
                k0 -= 80;

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);
                _sum1 = vmlaq_f32(_sum1, _k40, _r42);
                _sum1 = vmlaq_f32(_sum1, _k41, _r43);
                _sum1 = vmlaq_f32(_sum1, _k42, _r44);
                _sum1 = vmlaq_f32(_sum1, _k43, _r45);
                _sum1 = vmlaq_f32(_sum1, _k44, _r46);

                _mm256_storeu_ps(outptr0, _sum0);
                _mm256_storeu_ps(outptr0 + 4, _sum1);

                r0 += 4 * 4;
                r1 += 4 * 4;
                r2 += 4 * 4;
                r3 += 4 * 4;
                r4 += 4 * 4;
                outptr0 += 8;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;

                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 4);
                __m256 _r02 = _mm256_loadu_ps(r0 + 8);
                __m256 _r03 = _mm256_loadu_ps(r0 + 12);
                __m256 _r04 = _mm256_loadu_ps(r0 + 16);

                __m256 _k00 = _mm256_loadu_ps(k0);
                __m256 _k01 = _mm256_loadu_ps(k0 + 4);
                __m256 _k02 = _mm256_loadu_ps(k0 + 8);
                __m256 _k03 = _mm256_loadu_ps(k0 + 12);
                __m256 _k04 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 4);
                __m256 _r12 = _mm256_loadu_ps(r1 + 8);
                __m256 _r13 = _mm256_loadu_ps(r1 + 12);
                __m256 _r14 = _mm256_loadu_ps(r1 + 16);

                __m256 _k10 = _mm256_loadu_ps(k0);
                __m256 _k11 = _mm256_loadu_ps(k0 + 4);
                __m256 _k12 = _mm256_loadu_ps(k0 + 8);
                __m256 _k13 = _mm256_loadu_ps(k0 + 12);
                __m256 _k14 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);

                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 4);
                __m256 _r22 = _mm256_loadu_ps(r2 + 8);
                __m256 _r23 = _mm256_loadu_ps(r2 + 12);
                __m256 _r24 = _mm256_loadu_ps(r2 + 16);

                __m256 _k20 = _mm256_loadu_ps(k0);
                __m256 _k21 = _mm256_loadu_ps(k0 + 4);
                __m256 _k22 = _mm256_loadu_ps(k0 + 8);
                __m256 _k23 = _mm256_loadu_ps(k0 + 12);
                __m256 _k24 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);

                __m256 _r30 = _mm256_loadu_ps(r3);
                __m256 _r31 = _mm256_loadu_ps(r3 + 4);
                __m256 _r32 = _mm256_loadu_ps(r3 + 8);
                __m256 _r33 = _mm256_loadu_ps(r3 + 12);
                __m256 _r34 = _mm256_loadu_ps(r3 + 16);

                __m256 _k30 = _mm256_loadu_ps(k0);
                __m256 _k31 = _mm256_loadu_ps(k0 + 4);
                __m256 _k32 = _mm256_loadu_ps(k0 + 8);
                __m256 _k33 = _mm256_loadu_ps(k0 + 12);
                __m256 _k34 = _mm256_loadu_ps(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);

                __m256 _r40 = _mm256_loadu_ps(r4);
                __m256 _r41 = _mm256_loadu_ps(r4 + 4);
                __m256 _r42 = _mm256_loadu_ps(r4 + 8);
                __m256 _r43 = _mm256_loadu_ps(r4 + 12);
                __m256 _r44 = _mm256_loadu_ps(r4 + 16);

                __m256 _k40 = _mm256_loadu_ps(k0);
                __m256 _k41 = _mm256_loadu_ps(k0 + 4);
                __m256 _k42 = _mm256_loadu_ps(k0 + 8);
                __m256 _k43 = _mm256_loadu_ps(k0 + 12);
                __m256 _k44 = _mm256_loadu_ps(k0 + 16);
                k0 -= 80;

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);

                _mm256_storeu_ps(outptr0, _sum0);

                r0 += 2 * 4;
                r1 += 2 * 4;
                r2 += 2 * 4;
                r3 += 2 * 4;
                r4 += 2 * 4;
                outptr0 += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }
}

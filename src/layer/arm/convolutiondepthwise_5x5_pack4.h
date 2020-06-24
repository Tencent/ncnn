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

static void convdw5x5s1_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

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
                float32x4_t _sum00 = _bias0;
                float32x4_t _sum01 = _bias0;
                float32x4_t _sum02 = _bias0;
                float32x4_t _sum03 = _bias0;
                float32x4_t _sum10 = _bias0;
                float32x4_t _sum11 = _bias0;
                float32x4_t _sum12 = _bias0;
                float32x4_t _sum13 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);
                float32x4_t _r05 = vld1q_f32(r0 + 20);
                float32x4_t _r06 = vld1q_f32(r0 + 24);
                float32x4_t _r07 = vld1q_f32(r0 + 28);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
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
                _sum02 = vmlaq_f32(_sum02, _k00, _r02);
                _sum02 = vmlaq_f32(_sum02, _k01, _r03);
                _sum02 = vmlaq_f32(_sum02, _k02, _r04);
                _sum02 = vmlaq_f32(_sum02, _k03, _r05);
                _sum02 = vmlaq_f32(_sum02, _k04, _r06);
                _sum03 = vmlaq_f32(_sum03, _k00, _r03);
                _sum03 = vmlaq_f32(_sum03, _k01, _r04);
                _sum03 = vmlaq_f32(_sum03, _k02, _r05);
                _sum03 = vmlaq_f32(_sum03, _k03, _r06);
                _sum03 = vmlaq_f32(_sum03, _k04, _r07);

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);
                float32x4_t _r15 = vld1q_f32(r1 + 20);
                float32x4_t _r16 = vld1q_f32(r1 + 24);
                float32x4_t _r17 = vld1q_f32(r1 + 28);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
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
                _sum12 = vmlaq_f32(_sum12, _k00, _r12);
                _sum12 = vmlaq_f32(_sum12, _k01, _r13);
                _sum12 = vmlaq_f32(_sum12, _k02, _r14);
                _sum12 = vmlaq_f32(_sum12, _k03, _r15);
                _sum12 = vmlaq_f32(_sum12, _k04, _r16);
                _sum13 = vmlaq_f32(_sum13, _k00, _r13);
                _sum13 = vmlaq_f32(_sum13, _k01, _r14);
                _sum13 = vmlaq_f32(_sum13, _k02, _r15);
                _sum13 = vmlaq_f32(_sum13, _k03, _r16);
                _sum13 = vmlaq_f32(_sum13, _k04, _r17);

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
                _sum02 = vmlaq_f32(_sum02, _k10, _r12);
                _sum02 = vmlaq_f32(_sum02, _k11, _r13);
                _sum02 = vmlaq_f32(_sum02, _k12, _r14);
                _sum02 = vmlaq_f32(_sum02, _k13, _r15);
                _sum02 = vmlaq_f32(_sum02, _k14, _r16);
                _sum03 = vmlaq_f32(_sum03, _k10, _r13);
                _sum03 = vmlaq_f32(_sum03, _k11, _r14);
                _sum03 = vmlaq_f32(_sum03, _k12, _r15);
                _sum03 = vmlaq_f32(_sum03, _k13, _r16);
                _sum03 = vmlaq_f32(_sum03, _k14, _r17);

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);
                float32x4_t _r25 = vld1q_f32(r2 + 20);
                float32x4_t _r26 = vld1q_f32(r2 + 24);
                float32x4_t _r27 = vld1q_f32(r2 + 28);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
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
                _sum12 = vmlaq_f32(_sum12, _k10, _r22);
                _sum12 = vmlaq_f32(_sum12, _k11, _r23);
                _sum12 = vmlaq_f32(_sum12, _k12, _r24);
                _sum12 = vmlaq_f32(_sum12, _k13, _r25);
                _sum12 = vmlaq_f32(_sum12, _k14, _r26);
                _sum13 = vmlaq_f32(_sum13, _k10, _r23);
                _sum13 = vmlaq_f32(_sum13, _k11, _r24);
                _sum13 = vmlaq_f32(_sum13, _k12, _r25);
                _sum13 = vmlaq_f32(_sum13, _k13, _r26);
                _sum13 = vmlaq_f32(_sum13, _k14, _r27);

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
                _sum02 = vmlaq_f32(_sum02, _k20, _r22);
                _sum02 = vmlaq_f32(_sum02, _k21, _r23);
                _sum02 = vmlaq_f32(_sum02, _k22, _r24);
                _sum02 = vmlaq_f32(_sum02, _k23, _r25);
                _sum02 = vmlaq_f32(_sum02, _k24, _r26);
                _sum03 = vmlaq_f32(_sum03, _k20, _r23);
                _sum03 = vmlaq_f32(_sum03, _k21, _r24);
                _sum03 = vmlaq_f32(_sum03, _k22, _r25);
                _sum03 = vmlaq_f32(_sum03, _k23, _r26);
                _sum03 = vmlaq_f32(_sum03, _k24, _r27);

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);
                float32x4_t _r35 = vld1q_f32(r3 + 20);
                float32x4_t _r36 = vld1q_f32(r3 + 24);
                float32x4_t _r37 = vld1q_f32(r3 + 28);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
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
                _sum12 = vmlaq_f32(_sum12, _k20, _r32);
                _sum12 = vmlaq_f32(_sum12, _k21, _r33);
                _sum12 = vmlaq_f32(_sum12, _k22, _r34);
                _sum12 = vmlaq_f32(_sum12, _k23, _r35);
                _sum12 = vmlaq_f32(_sum12, _k24, _r36);
                _sum13 = vmlaq_f32(_sum13, _k20, _r33);
                _sum13 = vmlaq_f32(_sum13, _k21, _r34);
                _sum13 = vmlaq_f32(_sum13, _k22, _r35);
                _sum13 = vmlaq_f32(_sum13, _k23, _r36);
                _sum13 = vmlaq_f32(_sum13, _k24, _r37);

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
                _sum02 = vmlaq_f32(_sum02, _k30, _r32);
                _sum02 = vmlaq_f32(_sum02, _k31, _r33);
                _sum02 = vmlaq_f32(_sum02, _k32, _r34);
                _sum02 = vmlaq_f32(_sum02, _k33, _r35);
                _sum02 = vmlaq_f32(_sum02, _k34, _r36);
                _sum03 = vmlaq_f32(_sum03, _k30, _r33);
                _sum03 = vmlaq_f32(_sum03, _k31, _r34);
                _sum03 = vmlaq_f32(_sum03, _k32, _r35);
                _sum03 = vmlaq_f32(_sum03, _k33, _r36);
                _sum03 = vmlaq_f32(_sum03, _k34, _r37);

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);
                float32x4_t _r45 = vld1q_f32(r4 + 20);
                float32x4_t _r46 = vld1q_f32(r4 + 24);
                float32x4_t _r47 = vld1q_f32(r4 + 28);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
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
                _sum12 = vmlaq_f32(_sum12, _k30, _r42);
                _sum12 = vmlaq_f32(_sum12, _k31, _r43);
                _sum12 = vmlaq_f32(_sum12, _k32, _r44);
                _sum12 = vmlaq_f32(_sum12, _k33, _r45);
                _sum12 = vmlaq_f32(_sum12, _k34, _r46);
                _sum13 = vmlaq_f32(_sum13, _k30, _r43);
                _sum13 = vmlaq_f32(_sum13, _k31, _r44);
                _sum13 = vmlaq_f32(_sum13, _k32, _r45);
                _sum13 = vmlaq_f32(_sum13, _k33, _r46);
                _sum13 = vmlaq_f32(_sum13, _k34, _r47);

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
                _sum02 = vmlaq_f32(_sum02, _k40, _r42);
                _sum02 = vmlaq_f32(_sum02, _k41, _r43);
                _sum02 = vmlaq_f32(_sum02, _k42, _r44);
                _sum02 = vmlaq_f32(_sum02, _k43, _r45);
                _sum02 = vmlaq_f32(_sum02, _k44, _r46);
                _sum03 = vmlaq_f32(_sum03, _k40, _r43);
                _sum03 = vmlaq_f32(_sum03, _k41, _r44);
                _sum03 = vmlaq_f32(_sum03, _k42, _r45);
                _sum03 = vmlaq_f32(_sum03, _k43, _r46);
                _sum03 = vmlaq_f32(_sum03, _k44, _r47);

                float32x4_t _r50 = vld1q_f32(r5);
                float32x4_t _r51 = vld1q_f32(r5 + 4);
                float32x4_t _r52 = vld1q_f32(r5 + 8);
                float32x4_t _r53 = vld1q_f32(r5 + 12);
                float32x4_t _r54 = vld1q_f32(r5 + 16);
                float32x4_t _r55 = vld1q_f32(r5 + 20);
                float32x4_t _r56 = vld1q_f32(r5 + 24);
                float32x4_t _r57 = vld1q_f32(r5 + 28);

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
                _sum12 = vmlaq_f32(_sum12, _k40, _r52);
                _sum12 = vmlaq_f32(_sum12, _k41, _r53);
                _sum12 = vmlaq_f32(_sum12, _k42, _r54);
                _sum12 = vmlaq_f32(_sum12, _k43, _r55);
                _sum12 = vmlaq_f32(_sum12, _k44, _r56);
                _sum13 = vmlaq_f32(_sum13, _k40, _r53);
                _sum13 = vmlaq_f32(_sum13, _k41, _r54);
                _sum13 = vmlaq_f32(_sum13, _k42, _r55);
                _sum13 = vmlaq_f32(_sum13, _k43, _r56);
                _sum13 = vmlaq_f32(_sum13, _k44, _r57);

                vst1q_f32(outptr0, _sum00);
                vst1q_f32(outptr0 + 4, _sum01);
                vst1q_f32(outptr0 + 8, _sum02);
                vst1q_f32(outptr0 + 12, _sum03);
                vst1q_f32(outptr1, _sum10);
                vst1q_f32(outptr1 + 4, _sum11);
                vst1q_f32(outptr1 + 8, _sum12);
                vst1q_f32(outptr1 + 12, _sum13);

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
                float32x4_t _sum00 = _bias0;
                float32x4_t _sum01 = _bias0;
                float32x4_t _sum10 = _bias0;
                float32x4_t _sum11 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);
                float32x4_t _r05 = vld1q_f32(r0 + 20);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
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

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);
                float32x4_t _r15 = vld1q_f32(r1 + 20);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
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

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);
                float32x4_t _r25 = vld1q_f32(r2 + 20);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
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

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);
                float32x4_t _r35 = vld1q_f32(r3 + 20);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
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

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);
                float32x4_t _r45 = vld1q_f32(r4 + 20);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
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

                float32x4_t _r50 = vld1q_f32(r5);
                float32x4_t _r51 = vld1q_f32(r5 + 4);
                float32x4_t _r52 = vld1q_f32(r5 + 8);
                float32x4_t _r53 = vld1q_f32(r5 + 12);
                float32x4_t _r54 = vld1q_f32(r5 + 16);
                float32x4_t _r55 = vld1q_f32(r5 + 20);

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

                vst1q_f32(outptr0, _sum00);
                vst1q_f32(outptr0 + 4, _sum01);
                vst1q_f32(outptr1, _sum10);
                vst1q_f32(outptr1 + 4, _sum11);

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
                float32x4_t _sum0 = _bias0;
                float32x4_t _sum1 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
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

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
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

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
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

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
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

                float32x4_t _r50 = vld1q_f32(r5);
                float32x4_t _r51 = vld1q_f32(r5 + 4);
                float32x4_t _r52 = vld1q_f32(r5 + 8);
                float32x4_t _r53 = vld1q_f32(r5 + 12);
                float32x4_t _r54 = vld1q_f32(r5 + 16);

                _sum1 = vmlaq_f32(_sum1, _k40, _r50);
                _sum1 = vmlaq_f32(_sum1, _k41, _r51);
                _sum1 = vmlaq_f32(_sum1, _k42, _r52);
                _sum1 = vmlaq_f32(_sum1, _k43, _r53);
                _sum1 = vmlaq_f32(_sum1, _k44, _r54);

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr1, _sum1);

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
                float32x4_t _sum0 = _bias0;
                float32x4_t _sum1 = _bias0;
                float32x4_t _sum2 = _bias0;
                float32x4_t _sum3 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);
                float32x4_t _r05 = vld1q_f32(r0 + 20);
                float32x4_t _r06 = vld1q_f32(r0 + 24);
                float32x4_t _r07 = vld1q_f32(r0 + 28);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
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

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);
                float32x4_t _r15 = vld1q_f32(r1 + 20);
                float32x4_t _r16 = vld1q_f32(r1 + 24);
                float32x4_t _r17 = vld1q_f32(r1 + 28);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
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

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);
                float32x4_t _r25 = vld1q_f32(r2 + 20);
                float32x4_t _r26 = vld1q_f32(r2 + 24);
                float32x4_t _r27 = vld1q_f32(r2 + 28);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
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

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);
                float32x4_t _r35 = vld1q_f32(r3 + 20);
                float32x4_t _r36 = vld1q_f32(r3 + 24);
                float32x4_t _r37 = vld1q_f32(r3 + 28);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
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

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);
                float32x4_t _r45 = vld1q_f32(r4 + 20);
                float32x4_t _r46 = vld1q_f32(r4 + 24);
                float32x4_t _r47 = vld1q_f32(r4 + 28);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
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

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);
                vst1q_f32(outptr0 + 8, _sum2);
                vst1q_f32(outptr0 + 12, _sum3);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                r4 += 16;
                outptr0 += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                float32x4_t _sum0 = _bias0;
                float32x4_t _sum1 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);
                float32x4_t _r05 = vld1q_f32(r0 + 20);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
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

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);
                float32x4_t _r15 = vld1q_f32(r1 + 20);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
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

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);
                float32x4_t _r25 = vld1q_f32(r2 + 20);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
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

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);
                float32x4_t _r35 = vld1q_f32(r3 + 20);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
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

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);
                float32x4_t _r45 = vld1q_f32(r4 + 20);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
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

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                outptr0 += 8;
            }
            for (; j < outw; j++)
            {
                float32x4_t _sum0 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
                k0 -= 80;

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);

                vst1q_f32(outptr0, _sum0);

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

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

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
                float32x4_t _sum0 = _bias0;
                float32x4_t _sum1 = _bias0;
                float32x4_t _sum2 = _bias0;
                float32x4_t _sum3 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);
                float32x4_t _r05 = vld1q_f32(r0 + 20);
                float32x4_t _r06 = vld1q_f32(r0 + 24);
                float32x4_t _r07 = vld1q_f32(r0 + 28);
                float32x4_t _r08 = vld1q_f32(r0 + 32);
                float32x4_t _r09 = vld1q_f32(r0 + 36);
                float32x4_t _r010 = vld1q_f32(r0 + 40);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
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

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);
                float32x4_t _r15 = vld1q_f32(r1 + 20);
                float32x4_t _r16 = vld1q_f32(r1 + 24);
                float32x4_t _r17 = vld1q_f32(r1 + 28);
                float32x4_t _r18 = vld1q_f32(r1 + 32);
                float32x4_t _r19 = vld1q_f32(r1 + 36);
                float32x4_t _r110 = vld1q_f32(r1 + 40);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
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

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);
                float32x4_t _r25 = vld1q_f32(r2 + 20);
                float32x4_t _r26 = vld1q_f32(r2 + 24);
                float32x4_t _r27 = vld1q_f32(r2 + 28);
                float32x4_t _r28 = vld1q_f32(r2 + 32);
                float32x4_t _r29 = vld1q_f32(r2 + 36);
                float32x4_t _r210 = vld1q_f32(r2 + 40);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
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

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);
                float32x4_t _r35 = vld1q_f32(r3 + 20);
                float32x4_t _r36 = vld1q_f32(r3 + 24);
                float32x4_t _r37 = vld1q_f32(r3 + 28);
                float32x4_t _r38 = vld1q_f32(r3 + 32);
                float32x4_t _r39 = vld1q_f32(r3 + 36);
                float32x4_t _r310 = vld1q_f32(r3 + 40);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
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

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);
                float32x4_t _r45 = vld1q_f32(r4 + 20);
                float32x4_t _r46 = vld1q_f32(r4 + 24);
                float32x4_t _r47 = vld1q_f32(r4 + 28);
                float32x4_t _r48 = vld1q_f32(r4 + 32);
                float32x4_t _r49 = vld1q_f32(r4 + 36);
                float32x4_t _r410 = vld1q_f32(r4 + 40);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
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

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);
                vst1q_f32(outptr0 + 8, _sum2);
                vst1q_f32(outptr0 + 12, _sum3);

                r0 += 8 * 4;
                r1 += 8 * 4;
                r2 += 8 * 4;
                r3 += 8 * 4;
                r4 += 8 * 4;
                outptr0 += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                float32x4_t _sum0 = _bias0;
                float32x4_t _sum1 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);
                float32x4_t _r05 = vld1q_f32(r0 + 20);
                float32x4_t _r06 = vld1q_f32(r0 + 24);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
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

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);
                float32x4_t _r15 = vld1q_f32(r1 + 20);
                float32x4_t _r16 = vld1q_f32(r1 + 24);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
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

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);
                float32x4_t _r25 = vld1q_f32(r2 + 20);
                float32x4_t _r26 = vld1q_f32(r2 + 24);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
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

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);
                float32x4_t _r35 = vld1q_f32(r3 + 20);
                float32x4_t _r36 = vld1q_f32(r3 + 24);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
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

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);
                float32x4_t _r45 = vld1q_f32(r4 + 20);
                float32x4_t _r46 = vld1q_f32(r4 + 24);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
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

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);

                r0 += 4 * 4;
                r1 += 4 * 4;
                r2 += 4 * 4;
                r3 += 4 * 4;
                r4 += 4 * 4;
                outptr0 += 8;
            }
            for (; j < outw; j++)
            {
                float32x4_t _sum0 = _bias0;

                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r03 = vld1q_f32(r0 + 12);
                float32x4_t _r04 = vld1q_f32(r0 + 16);

                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k01 = vld1q_f32(k0 + 4);
                float32x4_t _k02 = vld1q_f32(k0 + 8);
                float32x4_t _k03 = vld1q_f32(k0 + 12);
                float32x4_t _k04 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r13 = vld1q_f32(r1 + 12);
                float32x4_t _r14 = vld1q_f32(r1 + 16);

                float32x4_t _k10 = vld1q_f32(k0);
                float32x4_t _k11 = vld1q_f32(k0 + 4);
                float32x4_t _k12 = vld1q_f32(k0 + 8);
                float32x4_t _k13 = vld1q_f32(k0 + 12);
                float32x4_t _k14 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                _sum0 = vmlaq_f32(_sum0, _k14, _r14);

                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);
                float32x4_t _r23 = vld1q_f32(r2 + 12);
                float32x4_t _r24 = vld1q_f32(r2 + 16);

                float32x4_t _k20 = vld1q_f32(k0);
                float32x4_t _k21 = vld1q_f32(k0 + 4);
                float32x4_t _k22 = vld1q_f32(k0 + 8);
                float32x4_t _k23 = vld1q_f32(k0 + 12);
                float32x4_t _k24 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                _sum0 = vmlaq_f32(_sum0, _k24, _r24);

                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r32 = vld1q_f32(r3 + 8);
                float32x4_t _r33 = vld1q_f32(r3 + 12);
                float32x4_t _r34 = vld1q_f32(r3 + 16);

                float32x4_t _k30 = vld1q_f32(k0);
                float32x4_t _k31 = vld1q_f32(k0 + 4);
                float32x4_t _k32 = vld1q_f32(k0 + 8);
                float32x4_t _k33 = vld1q_f32(k0 + 12);
                float32x4_t _k34 = vld1q_f32(k0 + 16);
                k0 += 20;

                _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                _sum0 = vmlaq_f32(_sum0, _k34, _r34);

                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r42 = vld1q_f32(r4 + 8);
                float32x4_t _r43 = vld1q_f32(r4 + 12);
                float32x4_t _r44 = vld1q_f32(r4 + 16);

                float32x4_t _k40 = vld1q_f32(k0);
                float32x4_t _k41 = vld1q_f32(k0 + 4);
                float32x4_t _k42 = vld1q_f32(k0 + 8);
                float32x4_t _k43 = vld1q_f32(k0 + 12);
                float32x4_t _k44 = vld1q_f32(k0 + 16);
                k0 -= 80;

                _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                _sum0 = vmlaq_f32(_sum0, _k44, _r44);

                vst1q_f32(outptr0, _sum0);

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

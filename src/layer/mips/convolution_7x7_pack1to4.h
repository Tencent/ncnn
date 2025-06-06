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

static void conv7x7s2_pack1to4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        v4f32 _bias0 = bias ? (v4f32)__msa_ld_w(bias + p * 4, 0) : (v4f32)__msa_fill_w(0);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);
            const float* r3 = img0.row(3);
            const float* r4 = img0.row(4);
            const float* r5 = img0.row(5);
            const float* r6 = img0.row(6);

            const float* kptr = kernel.channel(p).row(q);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__msa_ld_w(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__msa_ld_w(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__msa_ld_w(outptr0 + 4 * 3, 0);

                    v4f32 _k00 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k01 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k02 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k03 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k04 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k05 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k06 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4i32 _r0n = __msa_ld_w(r0 + 4, 0);
                    v4i32 _r0nn = __msa_ld_w(r0 + 8, 0);

                    v4f32 _r00 = (v4f32)__msa_splati_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__msa_splati_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__msa_splati_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__msa_splati_w(_r0, 3);
                    v4f32 _r04 = (v4f32)__msa_splati_w(_r0n, 0);
                    v4f32 _r05 = (v4f32)__msa_splati_w(_r0n, 1);
                    v4f32 _r06 = (v4f32)__msa_splati_w(_r0n, 2);
                    v4f32 _r07 = (v4f32)__msa_splati_w(_r0n, 3);
                    v4f32 _r08 = (v4f32)__msa_splati_w(_r0nn, 0);
                    v4f32 _r09 = (v4f32)__msa_splati_w(_r0nn, 1);
                    v4f32 _r0a = (v4f32)__msa_splati_w(_r0nn, 2);
                    v4f32 _r0b = (v4f32)__msa_splati_w(_r0nn, 3);
                    v4f32 _r0c = __msa_fill_w_f32(r0[12]);

                    _sum0 = __msa_fmadd_w(_sum0, _r00, _k00);
                    _sum1 = __msa_fmadd_w(_sum1, _r02, _k00);
                    _sum2 = __msa_fmadd_w(_sum2, _r04, _k00);
                    _sum3 = __msa_fmadd_w(_sum3, _r06, _k00);
                    _sum0 = __msa_fmadd_w(_sum0, _r01, _k01);
                    _sum1 = __msa_fmadd_w(_sum1, _r03, _k01);
                    _sum2 = __msa_fmadd_w(_sum2, _r05, _k01);
                    _sum3 = __msa_fmadd_w(_sum3, _r07, _k01);
                    _sum0 = __msa_fmadd_w(_sum0, _r02, _k02);
                    _sum1 = __msa_fmadd_w(_sum1, _r04, _k02);
                    _sum2 = __msa_fmadd_w(_sum2, _r06, _k02);
                    _sum3 = __msa_fmadd_w(_sum3, _r08, _k02);
                    _sum0 = __msa_fmadd_w(_sum0, _r03, _k03);
                    _sum1 = __msa_fmadd_w(_sum1, _r05, _k03);
                    _sum2 = __msa_fmadd_w(_sum2, _r07, _k03);
                    _sum3 = __msa_fmadd_w(_sum3, _r09, _k03);
                    _sum0 = __msa_fmadd_w(_sum0, _r04, _k04);
                    _sum1 = __msa_fmadd_w(_sum1, _r06, _k04);
                    _sum2 = __msa_fmadd_w(_sum2, _r08, _k04);
                    _sum3 = __msa_fmadd_w(_sum3, _r0a, _k04);
                    _sum0 = __msa_fmadd_w(_sum0, _r05, _k05);
                    _sum1 = __msa_fmadd_w(_sum1, _r07, _k05);
                    _sum2 = __msa_fmadd_w(_sum2, _r09, _k05);
                    _sum3 = __msa_fmadd_w(_sum3, _r0b, _k05);
                    _sum0 = __msa_fmadd_w(_sum0, _r06, _k06);
                    _sum1 = __msa_fmadd_w(_sum1, _r08, _k06);
                    _sum2 = __msa_fmadd_w(_sum2, _r0a, _k06);
                    _sum3 = __msa_fmadd_w(_sum3, _r0c, _k06);

                    v4f32 _k10 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k11 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k12 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k13 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k14 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k15 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k16 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4i32 _r1n = __msa_ld_w(r1 + 4, 0);
                    v4i32 _r1nn = __msa_ld_w(r1 + 8, 0);

                    v4f32 _r10 = (v4f32)__msa_splati_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__msa_splati_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__msa_splati_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__msa_splati_w(_r1, 3);
                    v4f32 _r14 = (v4f32)__msa_splati_w(_r1n, 0);
                    v4f32 _r15 = (v4f32)__msa_splati_w(_r1n, 1);
                    v4f32 _r16 = (v4f32)__msa_splati_w(_r1n, 2);
                    v4f32 _r17 = (v4f32)__msa_splati_w(_r1n, 3);
                    v4f32 _r18 = (v4f32)__msa_splati_w(_r1nn, 0);
                    v4f32 _r19 = (v4f32)__msa_splati_w(_r1nn, 1);
                    v4f32 _r1a = (v4f32)__msa_splati_w(_r1nn, 2);
                    v4f32 _r1b = (v4f32)__msa_splati_w(_r1nn, 3);
                    v4f32 _r1c = __msa_fill_w_f32(r1[12]);

                    _sum0 = __msa_fmadd_w(_sum0, _r10, _k10);
                    _sum1 = __msa_fmadd_w(_sum1, _r12, _k10);
                    _sum2 = __msa_fmadd_w(_sum2, _r14, _k10);
                    _sum3 = __msa_fmadd_w(_sum3, _r16, _k10);
                    _sum0 = __msa_fmadd_w(_sum0, _r11, _k11);
                    _sum1 = __msa_fmadd_w(_sum1, _r13, _k11);
                    _sum2 = __msa_fmadd_w(_sum2, _r15, _k11);
                    _sum3 = __msa_fmadd_w(_sum3, _r17, _k11);
                    _sum0 = __msa_fmadd_w(_sum0, _r12, _k12);
                    _sum1 = __msa_fmadd_w(_sum1, _r14, _k12);
                    _sum2 = __msa_fmadd_w(_sum2, _r16, _k12);
                    _sum3 = __msa_fmadd_w(_sum3, _r18, _k12);
                    _sum0 = __msa_fmadd_w(_sum0, _r13, _k13);
                    _sum1 = __msa_fmadd_w(_sum1, _r15, _k13);
                    _sum2 = __msa_fmadd_w(_sum2, _r17, _k13);
                    _sum3 = __msa_fmadd_w(_sum3, _r19, _k13);
                    _sum0 = __msa_fmadd_w(_sum0, _r14, _k14);
                    _sum1 = __msa_fmadd_w(_sum1, _r16, _k14);
                    _sum2 = __msa_fmadd_w(_sum2, _r18, _k14);
                    _sum3 = __msa_fmadd_w(_sum3, _r1a, _k14);
                    _sum0 = __msa_fmadd_w(_sum0, _r15, _k15);
                    _sum1 = __msa_fmadd_w(_sum1, _r17, _k15);
                    _sum2 = __msa_fmadd_w(_sum2, _r19, _k15);
                    _sum3 = __msa_fmadd_w(_sum3, _r1b, _k15);
                    _sum0 = __msa_fmadd_w(_sum0, _r16, _k16);
                    _sum1 = __msa_fmadd_w(_sum1, _r18, _k16);
                    _sum2 = __msa_fmadd_w(_sum2, _r1a, _k16);
                    _sum3 = __msa_fmadd_w(_sum3, _r1c, _k16);

                    v4f32 _k20 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k21 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k22 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k23 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k24 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k25 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k26 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4i32 _r2n = __msa_ld_w(r2 + 4, 0);
                    v4i32 _r2nn = __msa_ld_w(r2 + 8, 0);

                    v4f32 _r20 = (v4f32)__msa_splati_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__msa_splati_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__msa_splati_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__msa_splati_w(_r2, 3);
                    v4f32 _r24 = (v4f32)__msa_splati_w(_r2n, 0);
                    v4f32 _r25 = (v4f32)__msa_splati_w(_r2n, 1);
                    v4f32 _r26 = (v4f32)__msa_splati_w(_r2n, 2);
                    v4f32 _r27 = (v4f32)__msa_splati_w(_r2n, 3);
                    v4f32 _r28 = (v4f32)__msa_splati_w(_r2nn, 0);
                    v4f32 _r29 = (v4f32)__msa_splati_w(_r2nn, 1);
                    v4f32 _r2a = (v4f32)__msa_splati_w(_r2nn, 2);
                    v4f32 _r2b = (v4f32)__msa_splati_w(_r2nn, 3);
                    v4f32 _r2c = __msa_fill_w_f32(r2[12]);

                    _sum0 = __msa_fmadd_w(_sum0, _r20, _k20);
                    _sum1 = __msa_fmadd_w(_sum1, _r22, _k20);
                    _sum2 = __msa_fmadd_w(_sum2, _r24, _k20);
                    _sum3 = __msa_fmadd_w(_sum3, _r26, _k20);
                    _sum0 = __msa_fmadd_w(_sum0, _r21, _k21);
                    _sum1 = __msa_fmadd_w(_sum1, _r23, _k21);
                    _sum2 = __msa_fmadd_w(_sum2, _r25, _k21);
                    _sum3 = __msa_fmadd_w(_sum3, _r27, _k21);
                    _sum0 = __msa_fmadd_w(_sum0, _r22, _k22);
                    _sum1 = __msa_fmadd_w(_sum1, _r24, _k22);
                    _sum2 = __msa_fmadd_w(_sum2, _r26, _k22);
                    _sum3 = __msa_fmadd_w(_sum3, _r28, _k22);
                    _sum0 = __msa_fmadd_w(_sum0, _r23, _k23);
                    _sum1 = __msa_fmadd_w(_sum1, _r25, _k23);
                    _sum2 = __msa_fmadd_w(_sum2, _r27, _k23);
                    _sum3 = __msa_fmadd_w(_sum3, _r29, _k23);
                    _sum0 = __msa_fmadd_w(_sum0, _r24, _k24);
                    _sum1 = __msa_fmadd_w(_sum1, _r26, _k24);
                    _sum2 = __msa_fmadd_w(_sum2, _r28, _k24);
                    _sum3 = __msa_fmadd_w(_sum3, _r2a, _k24);
                    _sum0 = __msa_fmadd_w(_sum0, _r25, _k25);
                    _sum1 = __msa_fmadd_w(_sum1, _r27, _k25);
                    _sum2 = __msa_fmadd_w(_sum2, _r29, _k25);
                    _sum3 = __msa_fmadd_w(_sum3, _r2b, _k25);
                    _sum0 = __msa_fmadd_w(_sum0, _r26, _k26);
                    _sum1 = __msa_fmadd_w(_sum1, _r28, _k26);
                    _sum2 = __msa_fmadd_w(_sum2, _r2a, _k26);
                    _sum3 = __msa_fmadd_w(_sum3, _r2c, _k26);

                    v4f32 _k30 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k31 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k32 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k33 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k34 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k35 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k36 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r3 = __msa_ld_w(r3, 0);
                    v4i32 _r3n = __msa_ld_w(r3 + 4, 0);
                    v4i32 _r3nn = __msa_ld_w(r3 + 8, 0);

                    v4f32 _r30 = (v4f32)__msa_splati_w(_r3, 0);
                    v4f32 _r31 = (v4f32)__msa_splati_w(_r3, 1);
                    v4f32 _r32 = (v4f32)__msa_splati_w(_r3, 2);
                    v4f32 _r33 = (v4f32)__msa_splati_w(_r3, 3);
                    v4f32 _r34 = (v4f32)__msa_splati_w(_r3n, 0);
                    v4f32 _r35 = (v4f32)__msa_splati_w(_r3n, 1);
                    v4f32 _r36 = (v4f32)__msa_splati_w(_r3n, 2);
                    v4f32 _r37 = (v4f32)__msa_splati_w(_r3n, 3);
                    v4f32 _r38 = (v4f32)__msa_splati_w(_r3nn, 0);
                    v4f32 _r39 = (v4f32)__msa_splati_w(_r3nn, 1);
                    v4f32 _r3a = (v4f32)__msa_splati_w(_r3nn, 2);
                    v4f32 _r3b = (v4f32)__msa_splati_w(_r3nn, 3);
                    v4f32 _r3c = __msa_fill_w_f32(r3[12]);

                    _sum0 = __msa_fmadd_w(_sum0, _r30, _k30);
                    _sum1 = __msa_fmadd_w(_sum1, _r32, _k30);
                    _sum2 = __msa_fmadd_w(_sum2, _r34, _k30);
                    _sum3 = __msa_fmadd_w(_sum3, _r36, _k30);
                    _sum0 = __msa_fmadd_w(_sum0, _r31, _k31);
                    _sum1 = __msa_fmadd_w(_sum1, _r33, _k31);
                    _sum2 = __msa_fmadd_w(_sum2, _r35, _k31);
                    _sum3 = __msa_fmadd_w(_sum3, _r37, _k31);
                    _sum0 = __msa_fmadd_w(_sum0, _r32, _k32);
                    _sum1 = __msa_fmadd_w(_sum1, _r34, _k32);
                    _sum2 = __msa_fmadd_w(_sum2, _r36, _k32);
                    _sum3 = __msa_fmadd_w(_sum3, _r38, _k32);
                    _sum0 = __msa_fmadd_w(_sum0, _r33, _k33);
                    _sum1 = __msa_fmadd_w(_sum1, _r35, _k33);
                    _sum2 = __msa_fmadd_w(_sum2, _r37, _k33);
                    _sum3 = __msa_fmadd_w(_sum3, _r39, _k33);
                    _sum0 = __msa_fmadd_w(_sum0, _r34, _k34);
                    _sum1 = __msa_fmadd_w(_sum1, _r36, _k34);
                    _sum2 = __msa_fmadd_w(_sum2, _r38, _k34);
                    _sum3 = __msa_fmadd_w(_sum3, _r3a, _k34);
                    _sum0 = __msa_fmadd_w(_sum0, _r35, _k35);
                    _sum1 = __msa_fmadd_w(_sum1, _r37, _k35);
                    _sum2 = __msa_fmadd_w(_sum2, _r39, _k35);
                    _sum3 = __msa_fmadd_w(_sum3, _r3b, _k35);
                    _sum0 = __msa_fmadd_w(_sum0, _r36, _k36);
                    _sum1 = __msa_fmadd_w(_sum1, _r38, _k36);
                    _sum2 = __msa_fmadd_w(_sum2, _r3a, _k36);
                    _sum3 = __msa_fmadd_w(_sum3, _r3c, _k36);

                    v4f32 _k40 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k41 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k42 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k43 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k44 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k45 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k46 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r4 = __msa_ld_w(r4, 0);
                    v4i32 _r4n = __msa_ld_w(r4 + 4, 0);
                    v4i32 _r4nn = __msa_ld_w(r4 + 8, 0);

                    v4f32 _r40 = (v4f32)__msa_splati_w(_r4, 0);
                    v4f32 _r41 = (v4f32)__msa_splati_w(_r4, 1);
                    v4f32 _r42 = (v4f32)__msa_splati_w(_r4, 2);
                    v4f32 _r43 = (v4f32)__msa_splati_w(_r4, 3);
                    v4f32 _r44 = (v4f32)__msa_splati_w(_r4n, 0);
                    v4f32 _r45 = (v4f32)__msa_splati_w(_r4n, 1);
                    v4f32 _r46 = (v4f32)__msa_splati_w(_r4n, 2);
                    v4f32 _r47 = (v4f32)__msa_splati_w(_r4n, 3);
                    v4f32 _r48 = (v4f32)__msa_splati_w(_r4nn, 0);
                    v4f32 _r49 = (v4f32)__msa_splati_w(_r4nn, 1);
                    v4f32 _r4a = (v4f32)__msa_splati_w(_r4nn, 2);
                    v4f32 _r4b = (v4f32)__msa_splati_w(_r4nn, 3);
                    v4f32 _r4c = __msa_fill_w_f32(r4[12]);

                    _sum0 = __msa_fmadd_w(_sum0, _r40, _k40);
                    _sum1 = __msa_fmadd_w(_sum1, _r42, _k40);
                    _sum2 = __msa_fmadd_w(_sum2, _r44, _k40);
                    _sum3 = __msa_fmadd_w(_sum3, _r46, _k40);
                    _sum0 = __msa_fmadd_w(_sum0, _r41, _k41);
                    _sum1 = __msa_fmadd_w(_sum1, _r43, _k41);
                    _sum2 = __msa_fmadd_w(_sum2, _r45, _k41);
                    _sum3 = __msa_fmadd_w(_sum3, _r47, _k41);
                    _sum0 = __msa_fmadd_w(_sum0, _r42, _k42);
                    _sum1 = __msa_fmadd_w(_sum1, _r44, _k42);
                    _sum2 = __msa_fmadd_w(_sum2, _r46, _k42);
                    _sum3 = __msa_fmadd_w(_sum3, _r48, _k42);
                    _sum0 = __msa_fmadd_w(_sum0, _r43, _k43);
                    _sum1 = __msa_fmadd_w(_sum1, _r45, _k43);
                    _sum2 = __msa_fmadd_w(_sum2, _r47, _k43);
                    _sum3 = __msa_fmadd_w(_sum3, _r49, _k43);
                    _sum0 = __msa_fmadd_w(_sum0, _r44, _k44);
                    _sum1 = __msa_fmadd_w(_sum1, _r46, _k44);
                    _sum2 = __msa_fmadd_w(_sum2, _r48, _k44);
                    _sum3 = __msa_fmadd_w(_sum3, _r4a, _k44);
                    _sum0 = __msa_fmadd_w(_sum0, _r45, _k45);
                    _sum1 = __msa_fmadd_w(_sum1, _r47, _k45);
                    _sum2 = __msa_fmadd_w(_sum2, _r49, _k45);
                    _sum3 = __msa_fmadd_w(_sum3, _r4b, _k45);
                    _sum0 = __msa_fmadd_w(_sum0, _r46, _k46);
                    _sum1 = __msa_fmadd_w(_sum1, _r48, _k46);
                    _sum2 = __msa_fmadd_w(_sum2, _r4a, _k46);
                    _sum3 = __msa_fmadd_w(_sum3, _r4c, _k46);

                    v4f32 _k50 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k51 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k52 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k53 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k54 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k55 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k56 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r5 = __msa_ld_w(r5, 0);
                    v4i32 _r5n = __msa_ld_w(r5 + 4, 0);
                    v4i32 _r5nn = __msa_ld_w(r5 + 8, 0);

                    v4f32 _r50 = (v4f32)__msa_splati_w(_r5, 0);
                    v4f32 _r51 = (v4f32)__msa_splati_w(_r5, 1);
                    v4f32 _r52 = (v4f32)__msa_splati_w(_r5, 2);
                    v4f32 _r53 = (v4f32)__msa_splati_w(_r5, 3);
                    v4f32 _r54 = (v4f32)__msa_splati_w(_r5n, 0);
                    v4f32 _r55 = (v4f32)__msa_splati_w(_r5n, 1);
                    v4f32 _r56 = (v4f32)__msa_splati_w(_r5n, 2);
                    v4f32 _r57 = (v4f32)__msa_splati_w(_r5n, 3);
                    v4f32 _r58 = (v4f32)__msa_splati_w(_r5nn, 0);
                    v4f32 _r59 = (v4f32)__msa_splati_w(_r5nn, 1);
                    v4f32 _r5a = (v4f32)__msa_splati_w(_r5nn, 2);
                    v4f32 _r5b = (v4f32)__msa_splati_w(_r5nn, 3);
                    v4f32 _r5c = __msa_fill_w_f32(r5[12]);

                    _sum0 = __msa_fmadd_w(_sum0, _r50, _k50);
                    _sum1 = __msa_fmadd_w(_sum1, _r52, _k50);
                    _sum2 = __msa_fmadd_w(_sum2, _r54, _k50);
                    _sum3 = __msa_fmadd_w(_sum3, _r56, _k50);
                    _sum0 = __msa_fmadd_w(_sum0, _r51, _k51);
                    _sum1 = __msa_fmadd_w(_sum1, _r53, _k51);
                    _sum2 = __msa_fmadd_w(_sum2, _r55, _k51);
                    _sum3 = __msa_fmadd_w(_sum3, _r57, _k51);
                    _sum0 = __msa_fmadd_w(_sum0, _r52, _k52);
                    _sum1 = __msa_fmadd_w(_sum1, _r54, _k52);
                    _sum2 = __msa_fmadd_w(_sum2, _r56, _k52);
                    _sum3 = __msa_fmadd_w(_sum3, _r58, _k52);
                    _sum0 = __msa_fmadd_w(_sum0, _r53, _k53);
                    _sum1 = __msa_fmadd_w(_sum1, _r55, _k53);
                    _sum2 = __msa_fmadd_w(_sum2, _r57, _k53);
                    _sum3 = __msa_fmadd_w(_sum3, _r59, _k53);
                    _sum0 = __msa_fmadd_w(_sum0, _r54, _k54);
                    _sum1 = __msa_fmadd_w(_sum1, _r56, _k54);
                    _sum2 = __msa_fmadd_w(_sum2, _r58, _k54);
                    _sum3 = __msa_fmadd_w(_sum3, _r5a, _k54);
                    _sum0 = __msa_fmadd_w(_sum0, _r55, _k55);
                    _sum1 = __msa_fmadd_w(_sum1, _r57, _k55);
                    _sum2 = __msa_fmadd_w(_sum2, _r59, _k55);
                    _sum3 = __msa_fmadd_w(_sum3, _r5b, _k55);
                    _sum0 = __msa_fmadd_w(_sum0, _r56, _k56);
                    _sum1 = __msa_fmadd_w(_sum1, _r58, _k56);
                    _sum2 = __msa_fmadd_w(_sum2, _r5a, _k56);
                    _sum3 = __msa_fmadd_w(_sum3, _r5c, _k56);

                    v4f32 _k60 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k61 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k62 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k63 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k64 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k65 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k66 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr -= 4 * 42;

                    v4i32 _r6 = __msa_ld_w(r6, 0);
                    v4i32 _r6n = __msa_ld_w(r6 + 4, 0);
                    v4i32 _r6nn = __msa_ld_w(r6 + 8, 0);

                    v4f32 _r60 = (v4f32)__msa_splati_w(_r6, 0);
                    v4f32 _r61 = (v4f32)__msa_splati_w(_r6, 1);
                    v4f32 _r62 = (v4f32)__msa_splati_w(_r6, 2);
                    v4f32 _r63 = (v4f32)__msa_splati_w(_r6, 3);
                    v4f32 _r64 = (v4f32)__msa_splati_w(_r6n, 0);
                    v4f32 _r65 = (v4f32)__msa_splati_w(_r6n, 1);
                    v4f32 _r66 = (v4f32)__msa_splati_w(_r6n, 2);
                    v4f32 _r67 = (v4f32)__msa_splati_w(_r6n, 3);
                    v4f32 _r68 = (v4f32)__msa_splati_w(_r6nn, 0);
                    v4f32 _r69 = (v4f32)__msa_splati_w(_r6nn, 1);
                    v4f32 _r6a = (v4f32)__msa_splati_w(_r6nn, 2);
                    v4f32 _r6b = (v4f32)__msa_splati_w(_r6nn, 3);
                    v4f32 _r6c = __msa_fill_w_f32(r6[12]);

                    _sum0 = __msa_fmadd_w(_sum0, _r60, _k60);
                    _sum1 = __msa_fmadd_w(_sum1, _r62, _k60);
                    _sum2 = __msa_fmadd_w(_sum2, _r64, _k60);
                    _sum3 = __msa_fmadd_w(_sum3, _r66, _k60);
                    _sum0 = __msa_fmadd_w(_sum0, _r61, _k61);
                    _sum1 = __msa_fmadd_w(_sum1, _r63, _k61);
                    _sum2 = __msa_fmadd_w(_sum2, _r65, _k61);
                    _sum3 = __msa_fmadd_w(_sum3, _r67, _k61);
                    _sum0 = __msa_fmadd_w(_sum0, _r62, _k62);
                    _sum1 = __msa_fmadd_w(_sum1, _r64, _k62);
                    _sum2 = __msa_fmadd_w(_sum2, _r66, _k62);
                    _sum3 = __msa_fmadd_w(_sum3, _r68, _k62);
                    _sum0 = __msa_fmadd_w(_sum0, _r63, _k63);
                    _sum1 = __msa_fmadd_w(_sum1, _r65, _k63);
                    _sum2 = __msa_fmadd_w(_sum2, _r67, _k63);
                    _sum3 = __msa_fmadd_w(_sum3, _r69, _k63);
                    _sum0 = __msa_fmadd_w(_sum0, _r64, _k64);
                    _sum1 = __msa_fmadd_w(_sum1, _r66, _k64);
                    _sum2 = __msa_fmadd_w(_sum2, _r68, _k64);
                    _sum3 = __msa_fmadd_w(_sum3, _r6a, _k64);
                    _sum0 = __msa_fmadd_w(_sum0, _r65, _k65);
                    _sum1 = __msa_fmadd_w(_sum1, _r67, _k65);
                    _sum2 = __msa_fmadd_w(_sum2, _r69, _k65);
                    _sum3 = __msa_fmadd_w(_sum3, _r6b, _k65);
                    _sum0 = __msa_fmadd_w(_sum0, _r66, _k66);
                    _sum1 = __msa_fmadd_w(_sum1, _r68, _k66);
                    _sum2 = __msa_fmadd_w(_sum2, _r6a, _k66);
                    _sum3 = __msa_fmadd_w(_sum3, _r6c, _k66);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);

                    outptr0 += 4 * 4;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    r6 += 8;
                }
                for (; j < outw; j++)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);

                    v4f32 _k00 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k01 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k02 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k03 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k04 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k05 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k06 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4i32 _r0n = __msa_ld_w(r0 + 4, 0);

                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r0, 0), _k00);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r0, 1), _k01);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r0, 2), _k02);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r0, 3), _k03);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r0n, 0), _k04);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r0n, 1), _k05);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r0n, 2), _k06);

                    v4f32 _k10 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k11 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k12 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k13 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k14 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k15 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k16 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4i32 _r1n = __msa_ld_w(r1 + 4, 0);

                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r1, 0), _k10);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r1, 1), _k11);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r1, 2), _k12);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r1, 3), _k13);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r1n, 0), _k14);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r1n, 1), _k15);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r1n, 2), _k16);

                    v4f32 _k20 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k21 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k22 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k23 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k24 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k25 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k26 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4i32 _r2n = __msa_ld_w(r2 + 4, 0);

                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r2, 0), _k20);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r2, 1), _k21);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r2, 2), _k22);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r2, 3), _k23);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r2n, 0), _k24);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r2n, 1), _k25);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r2n, 2), _k26);

                    v4f32 _k30 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k31 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k32 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k33 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k34 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k35 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k36 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r3 = __msa_ld_w(r3, 0);
                    v4i32 _r3n = __msa_ld_w(r3 + 4, 0);

                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r3, 0), _k30);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r3, 1), _k31);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r3, 2), _k32);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r3, 3), _k33);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r3n, 0), _k34);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r3n, 1), _k35);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r3n, 2), _k36);

                    v4f32 _k40 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k41 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k42 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k43 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k44 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k45 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k46 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r4 = __msa_ld_w(r4, 0);
                    v4i32 _r4n = __msa_ld_w(r4 + 4, 0);

                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r4, 0), _k40);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r4, 1), _k41);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r4, 2), _k42);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r4, 3), _k43);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r4n, 0), _k44);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r4n, 1), _k45);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r4n, 2), _k46);

                    v4f32 _k50 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k51 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k52 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k53 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k54 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k55 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k56 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    v4i32 _r5 = __msa_ld_w(r5, 0);
                    v4i32 _r5n = __msa_ld_w(r5 + 4, 0);

                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r5, 0), _k50);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r5, 1), _k51);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r5, 2), _k52);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r5, 3), _k53);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r5n, 0), _k54);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r5n, 1), _k55);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r5n, 2), _k56);

                    v4f32 _k60 = (v4f32)__msa_ld_w(kptr, 0);
                    v4f32 _k61 = (v4f32)__msa_ld_w(kptr + 4, 0);
                    v4f32 _k62 = (v4f32)__msa_ld_w(kptr + 4 * 2, 0);
                    v4f32 _k63 = (v4f32)__msa_ld_w(kptr + 4 * 3, 0);
                    v4f32 _k64 = (v4f32)__msa_ld_w(kptr + 4 * 4, 0);
                    v4f32 _k65 = (v4f32)__msa_ld_w(kptr + 4 * 5, 0);
                    v4f32 _k66 = (v4f32)__msa_ld_w(kptr + 4 * 6, 0);

                    kptr -= 4 * 42;

                    v4i32 _r6 = __msa_ld_w(r6, 0);
                    v4i32 _r6n = __msa_ld_w(r6 + 4, 0);

                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r6, 0), _k60);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r6, 1), _k61);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r6, 2), _k62);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r6, 3), _k63);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r6n, 0), _k64);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r6n, 1), _k65);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_r6n, 2), _k66);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);

                    outptr0 += 4;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    r5 += 2;
                    r6 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
                r5 += tailstep;
                r6 += tailstep;
            }
        }
    }
}

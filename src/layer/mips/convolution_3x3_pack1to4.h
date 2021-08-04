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

static void conv3x3s1_pack1to4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        v4f32 _bias0 = bias ? (v4f32)__msa_ld_w(bias + p * 4, 0) : (v4f32)__msa_fill_w(0);
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

            v4f32 _k00 = (v4f32)__msa_ld_w(k0, 0);
            v4f32 _k01 = (v4f32)__msa_ld_w(k0 + 4, 0);
            v4f32 _k02 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
            v4f32 _k10 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
            v4f32 _k11 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
            v4f32 _k12 = (v4f32)__msa_ld_w(k0 + 4 * 5, 0);
            v4f32 _k20 = (v4f32)__msa_ld_w(k0 + 4 * 6, 0);
            v4f32 _k21 = (v4f32)__msa_ld_w(k0 + 4 * 7, 0);
            v4f32 _k22 = (v4f32)__msa_ld_w(k0 + 4 * 8, 0);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__msa_ld_w(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__msa_ld_w(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__msa_ld_w(outptr0 + 4 * 3, 0);
                    v4f32 _sum4 = (v4f32)__msa_ld_w(outptr0 + 4 * 4, 0);
                    v4f32 _sum5 = (v4f32)__msa_ld_w(outptr0 + 4 * 5, 0);
                    v4f32 _sum6 = (v4f32)__msa_ld_w(outptr0 + 4 * 6, 0);
                    v4f32 _sum7 = (v4f32)__msa_ld_w(outptr0 + 4 * 7, 0);

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

                    _sum0 = __msa_fmadd_w(_sum0, _r00, _k00);
                    _sum1 = __msa_fmadd_w(_sum1, _r01, _k00);
                    _sum2 = __msa_fmadd_w(_sum2, _r02, _k00);
                    _sum3 = __msa_fmadd_w(_sum3, _r03, _k00);
                    _sum4 = __msa_fmadd_w(_sum4, _r04, _k00);
                    _sum5 = __msa_fmadd_w(_sum5, _r05, _k00);
                    _sum6 = __msa_fmadd_w(_sum6, _r06, _k00);
                    _sum7 = __msa_fmadd_w(_sum7, _r07, _k00);
                    _sum0 = __msa_fmadd_w(_sum0, _r01, _k01);
                    _sum1 = __msa_fmadd_w(_sum1, _r02, _k01);
                    _sum2 = __msa_fmadd_w(_sum2, _r03, _k01);
                    _sum3 = __msa_fmadd_w(_sum3, _r04, _k01);
                    _sum4 = __msa_fmadd_w(_sum4, _r05, _k01);
                    _sum5 = __msa_fmadd_w(_sum5, _r06, _k01);
                    _sum6 = __msa_fmadd_w(_sum6, _r07, _k01);
                    _sum7 = __msa_fmadd_w(_sum7, _r08, _k01);
                    _sum0 = __msa_fmadd_w(_sum0, _r02, _k02);
                    _sum1 = __msa_fmadd_w(_sum1, _r03, _k02);
                    _sum2 = __msa_fmadd_w(_sum2, _r04, _k02);
                    _sum3 = __msa_fmadd_w(_sum3, _r05, _k02);
                    _sum4 = __msa_fmadd_w(_sum4, _r06, _k02);
                    _sum5 = __msa_fmadd_w(_sum5, _r07, _k02);
                    _sum6 = __msa_fmadd_w(_sum6, _r08, _k02);
                    _sum7 = __msa_fmadd_w(_sum7, _r09, _k02);

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

                    _sum0 = __msa_fmadd_w(_sum0, _r10, _k10);
                    _sum1 = __msa_fmadd_w(_sum1, _r11, _k10);
                    _sum2 = __msa_fmadd_w(_sum2, _r12, _k10);
                    _sum3 = __msa_fmadd_w(_sum3, _r13, _k10);
                    _sum4 = __msa_fmadd_w(_sum4, _r14, _k10);
                    _sum5 = __msa_fmadd_w(_sum5, _r15, _k10);
                    _sum6 = __msa_fmadd_w(_sum6, _r16, _k10);
                    _sum7 = __msa_fmadd_w(_sum7, _r17, _k10);
                    _sum0 = __msa_fmadd_w(_sum0, _r11, _k11);
                    _sum1 = __msa_fmadd_w(_sum1, _r12, _k11);
                    _sum2 = __msa_fmadd_w(_sum2, _r13, _k11);
                    _sum3 = __msa_fmadd_w(_sum3, _r14, _k11);
                    _sum4 = __msa_fmadd_w(_sum4, _r15, _k11);
                    _sum5 = __msa_fmadd_w(_sum5, _r16, _k11);
                    _sum6 = __msa_fmadd_w(_sum6, _r17, _k11);
                    _sum7 = __msa_fmadd_w(_sum7, _r18, _k11);
                    _sum0 = __msa_fmadd_w(_sum0, _r12, _k12);
                    _sum1 = __msa_fmadd_w(_sum1, _r13, _k12);
                    _sum2 = __msa_fmadd_w(_sum2, _r14, _k12);
                    _sum3 = __msa_fmadd_w(_sum3, _r15, _k12);
                    _sum4 = __msa_fmadd_w(_sum4, _r16, _k12);
                    _sum5 = __msa_fmadd_w(_sum5, _r17, _k12);
                    _sum6 = __msa_fmadd_w(_sum6, _r18, _k12);
                    _sum7 = __msa_fmadd_w(_sum7, _r19, _k12);

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

                    _sum0 = __msa_fmadd_w(_sum0, _r20, _k20);
                    _sum1 = __msa_fmadd_w(_sum1, _r21, _k20);
                    _sum2 = __msa_fmadd_w(_sum2, _r22, _k20);
                    _sum3 = __msa_fmadd_w(_sum3, _r23, _k20);
                    _sum4 = __msa_fmadd_w(_sum4, _r24, _k20);
                    _sum5 = __msa_fmadd_w(_sum5, _r25, _k20);
                    _sum6 = __msa_fmadd_w(_sum6, _r26, _k20);
                    _sum7 = __msa_fmadd_w(_sum7, _r27, _k20);
                    _sum0 = __msa_fmadd_w(_sum0, _r21, _k21);
                    _sum1 = __msa_fmadd_w(_sum1, _r22, _k21);
                    _sum2 = __msa_fmadd_w(_sum2, _r23, _k21);
                    _sum3 = __msa_fmadd_w(_sum3, _r24, _k21);
                    _sum4 = __msa_fmadd_w(_sum4, _r25, _k21);
                    _sum5 = __msa_fmadd_w(_sum5, _r26, _k21);
                    _sum6 = __msa_fmadd_w(_sum6, _r27, _k21);
                    _sum7 = __msa_fmadd_w(_sum7, _r28, _k21);
                    _sum0 = __msa_fmadd_w(_sum0, _r22, _k22);
                    _sum1 = __msa_fmadd_w(_sum1, _r23, _k22);
                    _sum2 = __msa_fmadd_w(_sum2, _r24, _k22);
                    _sum3 = __msa_fmadd_w(_sum3, _r25, _k22);
                    _sum4 = __msa_fmadd_w(_sum4, _r26, _k22);
                    _sum5 = __msa_fmadd_w(_sum5, _r27, _k22);
                    _sum6 = __msa_fmadd_w(_sum6, _r28, _k22);
                    _sum7 = __msa_fmadd_w(_sum7, _r29, _k22);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);
                    __msa_st_w((v4i32)_sum4, outptr0 + 4 * 4, 0);
                    __msa_st_w((v4i32)_sum5, outptr0 + 4 * 5, 0);
                    __msa_st_w((v4i32)_sum6, outptr0 + 4 * 6, 0);
                    __msa_st_w((v4i32)_sum7, outptr0 + 4 * 7, 0);

                    outptr0 += 4 * 8;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
                for (; j + 3 < outw; j += 4)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__msa_ld_w(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__msa_ld_w(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__msa_ld_w(outptr0 + 4 * 3, 0);

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4i32 _r0n = __msa_ld_w(r0 + 4, 0);

                    v4f32 _r00 = (v4f32)__msa_splati_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__msa_splati_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__msa_splati_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__msa_splati_w(_r0, 3);
                    v4f32 _r04 = (v4f32)__msa_splati_w(_r0n, 0);
                    v4f32 _r05 = (v4f32)__msa_splati_w(_r0n, 1);

                    _sum0 = __msa_fmadd_w(_sum0, _r00, _k00);
                    _sum1 = __msa_fmadd_w(_sum1, _r01, _k00);
                    _sum2 = __msa_fmadd_w(_sum2, _r02, _k00);
                    _sum3 = __msa_fmadd_w(_sum3, _r03, _k00);
                    _sum0 = __msa_fmadd_w(_sum0, _r01, _k01);
                    _sum1 = __msa_fmadd_w(_sum1, _r02, _k01);
                    _sum2 = __msa_fmadd_w(_sum2, _r03, _k01);
                    _sum3 = __msa_fmadd_w(_sum3, _r04, _k01);
                    _sum0 = __msa_fmadd_w(_sum0, _r02, _k02);
                    _sum1 = __msa_fmadd_w(_sum1, _r03, _k02);
                    _sum2 = __msa_fmadd_w(_sum2, _r04, _k02);
                    _sum3 = __msa_fmadd_w(_sum3, _r05, _k02);

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4i32 _r1n = __msa_ld_w(r1 + 4, 0);

                    v4f32 _r10 = (v4f32)__msa_splati_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__msa_splati_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__msa_splati_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__msa_splati_w(_r1, 3);
                    v4f32 _r14 = (v4f32)__msa_splati_w(_r1n, 0);
                    v4f32 _r15 = (v4f32)__msa_splati_w(_r1n, 1);

                    _sum0 = __msa_fmadd_w(_sum0, _r10, _k10);
                    _sum1 = __msa_fmadd_w(_sum1, _r11, _k10);
                    _sum2 = __msa_fmadd_w(_sum2, _r12, _k10);
                    _sum3 = __msa_fmadd_w(_sum3, _r13, _k10);
                    _sum0 = __msa_fmadd_w(_sum0, _r11, _k11);
                    _sum1 = __msa_fmadd_w(_sum1, _r12, _k11);
                    _sum2 = __msa_fmadd_w(_sum2, _r13, _k11);
                    _sum3 = __msa_fmadd_w(_sum3, _r14, _k11);
                    _sum0 = __msa_fmadd_w(_sum0, _r12, _k12);
                    _sum1 = __msa_fmadd_w(_sum1, _r13, _k12);
                    _sum2 = __msa_fmadd_w(_sum2, _r14, _k12);
                    _sum3 = __msa_fmadd_w(_sum3, _r15, _k12);

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4i32 _r2n = __msa_ld_w(r2 + 4, 0);

                    v4f32 _r20 = (v4f32)__msa_splati_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__msa_splati_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__msa_splati_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__msa_splati_w(_r2, 3);
                    v4f32 _r24 = (v4f32)__msa_splati_w(_r2n, 0);
                    v4f32 _r25 = (v4f32)__msa_splati_w(_r2n, 1);

                    _sum0 = __msa_fmadd_w(_sum0, _r20, _k20);
                    _sum1 = __msa_fmadd_w(_sum1, _r21, _k20);
                    _sum2 = __msa_fmadd_w(_sum2, _r22, _k20);
                    _sum3 = __msa_fmadd_w(_sum3, _r23, _k20);
                    _sum0 = __msa_fmadd_w(_sum0, _r21, _k21);
                    _sum1 = __msa_fmadd_w(_sum1, _r22, _k21);
                    _sum2 = __msa_fmadd_w(_sum2, _r23, _k21);
                    _sum3 = __msa_fmadd_w(_sum3, _r24, _k21);
                    _sum0 = __msa_fmadd_w(_sum0, _r22, _k22);
                    _sum1 = __msa_fmadd_w(_sum1, _r23, _k22);
                    _sum2 = __msa_fmadd_w(_sum2, _r24, _k22);
                    _sum3 = __msa_fmadd_w(_sum3, _r25, _k22);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);

                    outptr0 += 4 * 4;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                for (; j + 1 < outw; j += 2)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__msa_ld_w(outptr0 + 4, 0);

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4f32 _r00 = (v4f32)__msa_splati_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__msa_splati_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__msa_splati_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__msa_splati_w(_r0, 3);

                    _sum0 = __msa_fmadd_w(_sum0, _r00, _k00);
                    _sum1 = __msa_fmadd_w(_sum1, _r01, _k00);
                    _sum0 = __msa_fmadd_w(_sum0, _r01, _k01);
                    _sum1 = __msa_fmadd_w(_sum1, _r02, _k01);
                    _sum0 = __msa_fmadd_w(_sum0, _r02, _k02);
                    _sum1 = __msa_fmadd_w(_sum1, _r03, _k02);

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4f32 _r10 = (v4f32)__msa_splati_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__msa_splati_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__msa_splati_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__msa_splati_w(_r1, 3);

                    _sum0 = __msa_fmadd_w(_sum0, _r10, _k10);
                    _sum1 = __msa_fmadd_w(_sum1, _r11, _k10);
                    _sum0 = __msa_fmadd_w(_sum0, _r11, _k11);
                    _sum1 = __msa_fmadd_w(_sum1, _r12, _k11);
                    _sum0 = __msa_fmadd_w(_sum0, _r12, _k12);
                    _sum1 = __msa_fmadd_w(_sum1, _r13, _k12);

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4f32 _r20 = (v4f32)__msa_splati_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__msa_splati_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__msa_splati_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__msa_splati_w(_r2, 3);

                    _sum0 = __msa_fmadd_w(_sum0, _r20, _k20);
                    _sum1 = __msa_fmadd_w(_sum1, _r21, _k20);
                    _sum0 = __msa_fmadd_w(_sum0, _r21, _k21);
                    _sum1 = __msa_fmadd_w(_sum1, _r22, _k21);
                    _sum0 = __msa_fmadd_w(_sum0, _r22, _k22);
                    _sum1 = __msa_fmadd_w(_sum1, _r23, _k22);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);

                    outptr0 += 4 * 2;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }
                for (; j < outw; j++)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4f32 _r00 = (v4f32)__msa_splati_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__msa_splati_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__msa_splati_w(_r0, 2);

                    _sum0 = __msa_fmadd_w(_sum0, _r00, _k00);
                    _sum0 = __msa_fmadd_w(_sum0, _r01, _k01);
                    _sum0 = __msa_fmadd_w(_sum0, _r02, _k02);

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4f32 _r10 = (v4f32)__msa_splati_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__msa_splati_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__msa_splati_w(_r1, 2);

                    _sum0 = __msa_fmadd_w(_sum0, _r10, _k10);
                    _sum0 = __msa_fmadd_w(_sum0, _r11, _k11);
                    _sum0 = __msa_fmadd_w(_sum0, _r12, _k12);

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4f32 _r20 = (v4f32)__msa_splati_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__msa_splati_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__msa_splati_w(_r2, 2);

                    _sum0 = __msa_fmadd_w(_sum0, _r20, _k20);
                    _sum0 = __msa_fmadd_w(_sum0, _r21, _k21);
                    _sum0 = __msa_fmadd_w(_sum0, _r22, _k22);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);

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

static void conv3x3s2_pack1to4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        const float* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            float* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            v4f32 _k00 = (v4f32)__msa_ld_w(k0, 0);
            v4f32 _k01 = (v4f32)__msa_ld_w(k0 + 4, 0);
            v4f32 _k02 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
            v4f32 _k10 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
            v4f32 _k11 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
            v4f32 _k12 = (v4f32)__msa_ld_w(k0 + 4 * 5, 0);
            v4f32 _k20 = (v4f32)__msa_ld_w(k0 + 4 * 6, 0);
            v4f32 _k21 = (v4f32)__msa_ld_w(k0 + 4 * 7, 0);
            v4f32 _k22 = (v4f32)__msa_ld_w(k0 + 4 * 8, 0);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__msa_ld_w(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__msa_ld_w(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__msa_ld_w(outptr0 + 4 * 3, 0);
                    v4f32 _sum4 = (v4f32)__msa_ld_w(outptr0 + 4 * 4, 0);
                    v4f32 _sum5 = (v4f32)__msa_ld_w(outptr0 + 4 * 5, 0);
                    v4f32 _sum6 = (v4f32)__msa_ld_w(outptr0 + 4 * 6, 0);
                    v4f32 _sum7 = (v4f32)__msa_ld_w(outptr0 + 4 * 7, 0);

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4i32 _r0n = __msa_ld_w(r0 + 4, 0);
                    v4i32 _r0nn = __msa_ld_w(r0 + 8, 0);
                    v4i32 _r0nnn = __msa_ld_w(r0 + 12, 0);

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
                    v4f32 _r0c = (v4f32)__msa_splati_w(_r0nnn, 0);
                    v4f32 _r0d = (v4f32)__msa_splati_w(_r0nnn, 1);
                    v4f32 _r0e = (v4f32)__msa_splati_w(_r0nnn, 2);
                    v4f32 _r0f = (v4f32)__msa_splati_w(_r0nnn, 3);
                    v4f32 _r0g = __msa_fill_w_f32(r0[16]);

                    _sum0 = __msa_fmadd_w(_sum0, _r00, _k00);
                    _sum1 = __msa_fmadd_w(_sum1, _r02, _k00);
                    _sum2 = __msa_fmadd_w(_sum2, _r04, _k00);
                    _sum3 = __msa_fmadd_w(_sum3, _r06, _k00);
                    _sum4 = __msa_fmadd_w(_sum4, _r08, _k00);
                    _sum5 = __msa_fmadd_w(_sum5, _r0a, _k00);
                    _sum6 = __msa_fmadd_w(_sum6, _r0c, _k00);
                    _sum7 = __msa_fmadd_w(_sum7, _r0e, _k00);
                    _sum0 = __msa_fmadd_w(_sum0, _r01, _k01);
                    _sum1 = __msa_fmadd_w(_sum1, _r03, _k01);
                    _sum2 = __msa_fmadd_w(_sum2, _r05, _k01);
                    _sum3 = __msa_fmadd_w(_sum3, _r07, _k01);
                    _sum4 = __msa_fmadd_w(_sum4, _r09, _k01);
                    _sum5 = __msa_fmadd_w(_sum5, _r0b, _k01);
                    _sum6 = __msa_fmadd_w(_sum6, _r0d, _k01);
                    _sum7 = __msa_fmadd_w(_sum7, _r0f, _k01);
                    _sum0 = __msa_fmadd_w(_sum0, _r02, _k02);
                    _sum1 = __msa_fmadd_w(_sum1, _r04, _k02);
                    _sum2 = __msa_fmadd_w(_sum2, _r06, _k02);
                    _sum3 = __msa_fmadd_w(_sum3, _r08, _k02);
                    _sum4 = __msa_fmadd_w(_sum4, _r0a, _k02);
                    _sum5 = __msa_fmadd_w(_sum5, _r0c, _k02);
                    _sum6 = __msa_fmadd_w(_sum6, _r0e, _k02);
                    _sum7 = __msa_fmadd_w(_sum7, _r0g, _k02);

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4i32 _r1n = __msa_ld_w(r1 + 4, 0);
                    v4i32 _r1nn = __msa_ld_w(r1 + 8, 0);
                    v4i32 _r1nnn = __msa_ld_w(r1 + 12, 0);

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
                    v4f32 _r1c = (v4f32)__msa_splati_w(_r1nnn, 0);
                    v4f32 _r1d = (v4f32)__msa_splati_w(_r1nnn, 1);
                    v4f32 _r1e = (v4f32)__msa_splati_w(_r1nnn, 2);
                    v4f32 _r1f = (v4f32)__msa_splati_w(_r1nnn, 3);
                    v4f32 _r1g = __msa_fill_w_f32(r1[16]);

                    _sum0 = __msa_fmadd_w(_sum0, _r10, _k10);
                    _sum1 = __msa_fmadd_w(_sum1, _r12, _k10);
                    _sum2 = __msa_fmadd_w(_sum2, _r14, _k10);
                    _sum3 = __msa_fmadd_w(_sum3, _r16, _k10);
                    _sum4 = __msa_fmadd_w(_sum4, _r18, _k10);
                    _sum5 = __msa_fmadd_w(_sum5, _r1a, _k10);
                    _sum6 = __msa_fmadd_w(_sum6, _r1c, _k10);
                    _sum7 = __msa_fmadd_w(_sum7, _r1e, _k10);
                    _sum0 = __msa_fmadd_w(_sum0, _r11, _k11);
                    _sum1 = __msa_fmadd_w(_sum1, _r13, _k11);
                    _sum2 = __msa_fmadd_w(_sum2, _r15, _k11);
                    _sum3 = __msa_fmadd_w(_sum3, _r17, _k11);
                    _sum4 = __msa_fmadd_w(_sum4, _r19, _k11);
                    _sum5 = __msa_fmadd_w(_sum5, _r1b, _k11);
                    _sum6 = __msa_fmadd_w(_sum6, _r1d, _k11);
                    _sum7 = __msa_fmadd_w(_sum7, _r1f, _k11);
                    _sum0 = __msa_fmadd_w(_sum0, _r12, _k12);
                    _sum1 = __msa_fmadd_w(_sum1, _r14, _k12);
                    _sum2 = __msa_fmadd_w(_sum2, _r16, _k12);
                    _sum3 = __msa_fmadd_w(_sum3, _r18, _k12);
                    _sum4 = __msa_fmadd_w(_sum4, _r1a, _k12);
                    _sum5 = __msa_fmadd_w(_sum5, _r1c, _k12);
                    _sum6 = __msa_fmadd_w(_sum6, _r1e, _k12);
                    _sum7 = __msa_fmadd_w(_sum7, _r1g, _k12);

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4i32 _r2n = __msa_ld_w(r2 + 4, 0);
                    v4i32 _r2nn = __msa_ld_w(r2 + 8, 0);
                    v4i32 _r2nnn = __msa_ld_w(r2 + 12, 0);

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
                    v4f32 _r2c = (v4f32)__msa_splati_w(_r2nnn, 0);
                    v4f32 _r2d = (v4f32)__msa_splati_w(_r2nnn, 1);
                    v4f32 _r2e = (v4f32)__msa_splati_w(_r2nnn, 2);
                    v4f32 _r2f = (v4f32)__msa_splati_w(_r2nnn, 3);
                    v4f32 _r2g = __msa_fill_w_f32(r2[16]);

                    _sum0 = __msa_fmadd_w(_sum0, _r20, _k20);
                    _sum1 = __msa_fmadd_w(_sum1, _r22, _k20);
                    _sum2 = __msa_fmadd_w(_sum2, _r24, _k20);
                    _sum3 = __msa_fmadd_w(_sum3, _r26, _k20);
                    _sum4 = __msa_fmadd_w(_sum4, _r28, _k20);
                    _sum5 = __msa_fmadd_w(_sum5, _r2a, _k20);
                    _sum6 = __msa_fmadd_w(_sum6, _r2c, _k20);
                    _sum7 = __msa_fmadd_w(_sum7, _r2e, _k20);
                    _sum0 = __msa_fmadd_w(_sum0, _r21, _k21);
                    _sum1 = __msa_fmadd_w(_sum1, _r23, _k21);
                    _sum2 = __msa_fmadd_w(_sum2, _r25, _k21);
                    _sum3 = __msa_fmadd_w(_sum3, _r27, _k21);
                    _sum4 = __msa_fmadd_w(_sum4, _r29, _k21);
                    _sum5 = __msa_fmadd_w(_sum5, _r2b, _k21);
                    _sum6 = __msa_fmadd_w(_sum6, _r2d, _k21);
                    _sum7 = __msa_fmadd_w(_sum7, _r2f, _k21);
                    _sum0 = __msa_fmadd_w(_sum0, _r22, _k22);
                    _sum1 = __msa_fmadd_w(_sum1, _r24, _k22);
                    _sum2 = __msa_fmadd_w(_sum2, _r26, _k22);
                    _sum3 = __msa_fmadd_w(_sum3, _r28, _k22);
                    _sum4 = __msa_fmadd_w(_sum4, _r2a, _k22);
                    _sum5 = __msa_fmadd_w(_sum5, _r2c, _k22);
                    _sum6 = __msa_fmadd_w(_sum6, _r2e, _k22);
                    _sum7 = __msa_fmadd_w(_sum7, _r2g, _k22);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);
                    __msa_st_w((v4i32)_sum4, outptr0 + 4 * 4, 0);
                    __msa_st_w((v4i32)_sum5, outptr0 + 4 * 5, 0);
                    __msa_st_w((v4i32)_sum6, outptr0 + 4 * 6, 0);
                    __msa_st_w((v4i32)_sum7, outptr0 + 4 * 7, 0);

                    outptr0 += 4 * 8;

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                }
                for (; j + 3 < outw; j += 4)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__msa_ld_w(outptr0 + 4, 0);
                    v4f32 _sum2 = (v4f32)__msa_ld_w(outptr0 + 4 * 2, 0);
                    v4f32 _sum3 = (v4f32)__msa_ld_w(outptr0 + 4 * 3, 0);

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4i32 _r0n = __msa_ld_w(r0 + 4, 0);

                    v4f32 _r00 = (v4f32)__msa_splati_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__msa_splati_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__msa_splati_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__msa_splati_w(_r0, 3);
                    v4f32 _r04 = (v4f32)__msa_splati_w(_r0n, 0);
                    v4f32 _r05 = (v4f32)__msa_splati_w(_r0n, 1);
                    v4f32 _r06 = (v4f32)__msa_splati_w(_r0n, 2);
                    v4f32 _r07 = (v4f32)__msa_splati_w(_r0n, 3);
                    v4f32 _r08 = __msa_fill_w_f32(r0[8]);

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

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4i32 _r1n = __msa_ld_w(r1 + 4, 0);

                    v4f32 _r10 = (v4f32)__msa_splati_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__msa_splati_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__msa_splati_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__msa_splati_w(_r1, 3);
                    v4f32 _r14 = (v4f32)__msa_splati_w(_r1n, 0);
                    v4f32 _r15 = (v4f32)__msa_splati_w(_r1n, 1);
                    v4f32 _r16 = (v4f32)__msa_splati_w(_r1n, 2);
                    v4f32 _r17 = (v4f32)__msa_splati_w(_r1n, 3);
                    v4f32 _r18 = __msa_fill_w_f32(r1[8]);

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

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4i32 _r2n = __msa_ld_w(r2 + 4, 0);

                    v4f32 _r20 = (v4f32)__msa_splati_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__msa_splati_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__msa_splati_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__msa_splati_w(_r2, 3);
                    v4f32 _r24 = (v4f32)__msa_splati_w(_r2n, 0);
                    v4f32 _r25 = (v4f32)__msa_splati_w(_r2n, 1);
                    v4f32 _r26 = (v4f32)__msa_splati_w(_r2n, 2);
                    v4f32 _r27 = (v4f32)__msa_splati_w(_r2n, 3);
                    v4f32 _r28 = __msa_fill_w_f32(r2[8]);

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

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);

                    outptr0 += 4 * 4;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
                for (; j + 1 < outw; j += 2)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);
                    v4f32 _sum1 = (v4f32)__msa_ld_w(outptr0 + 4, 0);

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4f32 _r00 = (v4f32)__msa_splati_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__msa_splati_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__msa_splati_w(_r0, 2);
                    v4f32 _r03 = (v4f32)__msa_splati_w(_r0, 3);
                    v4f32 _r04 = __msa_fill_w_f32(r0[4]);

                    _sum0 = __msa_fmadd_w(_sum0, _r00, _k00);
                    _sum1 = __msa_fmadd_w(_sum1, _r02, _k00);
                    _sum0 = __msa_fmadd_w(_sum0, _r01, _k01);
                    _sum1 = __msa_fmadd_w(_sum1, _r03, _k01);
                    _sum0 = __msa_fmadd_w(_sum0, _r02, _k02);
                    _sum1 = __msa_fmadd_w(_sum1, _r04, _k02);

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4f32 _r10 = (v4f32)__msa_splati_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__msa_splati_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__msa_splati_w(_r1, 2);
                    v4f32 _r13 = (v4f32)__msa_splati_w(_r1, 3);
                    v4f32 _r14 = __msa_fill_w_f32(r1[4]);

                    _sum0 = __msa_fmadd_w(_sum0, _r10, _k10);
                    _sum1 = __msa_fmadd_w(_sum1, _r12, _k10);
                    _sum0 = __msa_fmadd_w(_sum0, _r11, _k11);
                    _sum1 = __msa_fmadd_w(_sum1, _r13, _k11);
                    _sum0 = __msa_fmadd_w(_sum0, _r12, _k12);
                    _sum1 = __msa_fmadd_w(_sum1, _r14, _k12);

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4f32 _r20 = (v4f32)__msa_splati_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__msa_splati_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__msa_splati_w(_r2, 2);
                    v4f32 _r23 = (v4f32)__msa_splati_w(_r2, 3);
                    v4f32 _r24 = __msa_fill_w_f32(r2[4]);

                    _sum0 = __msa_fmadd_w(_sum0, _r20, _k20);
                    _sum1 = __msa_fmadd_w(_sum1, _r22, _k20);
                    _sum0 = __msa_fmadd_w(_sum0, _r21, _k21);
                    _sum1 = __msa_fmadd_w(_sum1, _r23, _k21);
                    _sum0 = __msa_fmadd_w(_sum0, _r22, _k22);
                    _sum1 = __msa_fmadd_w(_sum1, _r24, _k22);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);

                    outptr0 += 4 * 2;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                for (; j < outw; j++)
                {
                    v4f32 _sum0 = (v4f32)__msa_ld_w(outptr0, 0);

                    v4i32 _r0 = __msa_ld_w(r0, 0);
                    v4f32 _r00 = (v4f32)__msa_splati_w(_r0, 0);
                    v4f32 _r01 = (v4f32)__msa_splati_w(_r0, 1);
                    v4f32 _r02 = (v4f32)__msa_splati_w(_r0, 2);

                    _sum0 = __msa_fmadd_w(_sum0, _r00, _k00);
                    _sum0 = __msa_fmadd_w(_sum0, _r01, _k01);
                    _sum0 = __msa_fmadd_w(_sum0, _r02, _k02);

                    v4i32 _r1 = __msa_ld_w(r1, 0);
                    v4f32 _r10 = (v4f32)__msa_splati_w(_r1, 0);
                    v4f32 _r11 = (v4f32)__msa_splati_w(_r1, 1);
                    v4f32 _r12 = (v4f32)__msa_splati_w(_r1, 2);

                    _sum0 = __msa_fmadd_w(_sum0, _r10, _k10);
                    _sum0 = __msa_fmadd_w(_sum0, _r11, _k11);
                    _sum0 = __msa_fmadd_w(_sum0, _r12, _k12);

                    v4i32 _r2 = __msa_ld_w(r2, 0);
                    v4f32 _r20 = (v4f32)__msa_splati_w(_r2, 0);
                    v4f32 _r21 = (v4f32)__msa_splati_w(_r2, 1);
                    v4f32 _r22 = (v4f32)__msa_splati_w(_r2, 2);

                    _sum0 = __msa_fmadd_w(_sum0, _r20, _k20);
                    _sum0 = __msa_fmadd_w(_sum0, _r21, _k21);
                    _sum0 = __msa_fmadd_w(_sum0, _r22, _k22);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);

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

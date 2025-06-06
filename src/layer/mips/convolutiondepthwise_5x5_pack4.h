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

static void convdw5x5s1_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        v4f32 _bias0 = bias ? (v4f32)__msa_ld_w(bias + g * 4, 0) : (v4f32)__msa_fill_w(0);

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
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j < outw; j++)
            {
                __builtin_prefetch(r0 + 16);
                __builtin_prefetch(r1 + 16);
                __builtin_prefetch(r2 + 16);
                __builtin_prefetch(r3 + 16);
                __builtin_prefetch(r4 + 16);
                __builtin_prefetch(r5 + 16);

                __builtin_prefetch(k0 + 400);

                v4f32 _sum0 = _bias0;
                v4f32 _sum1 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);
                v4f32 _r04 = (v4f32)__msa_ld_w(r0 + 4 * 4, 0);

                v4f32 _k00 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k01 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k02 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k03 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k04 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k00, _r00);
                _sum0 = __msa_fmadd_w(_sum0, _k01, _r01);
                _sum0 = __msa_fmadd_w(_sum0, _k02, _r02);
                _sum0 = __msa_fmadd_w(_sum0, _k03, _r03);
                _sum0 = __msa_fmadd_w(_sum0, _k04, _r04);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);
                v4f32 _r13 = (v4f32)__msa_ld_w(r1 + 4 * 3, 0);
                v4f32 _r14 = (v4f32)__msa_ld_w(r1 + 4 * 4, 0);

                _sum1 = __msa_fmadd_w(_sum1, _k00, _r10);
                _sum1 = __msa_fmadd_w(_sum1, _k01, _r11);
                _sum1 = __msa_fmadd_w(_sum1, _k02, _r12);
                _sum1 = __msa_fmadd_w(_sum1, _k03, _r13);
                _sum1 = __msa_fmadd_w(_sum1, _k04, _r14);

                v4f32 _k10 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k11 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k12 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k13 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k14 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k10, _r10);
                _sum0 = __msa_fmadd_w(_sum0, _k11, _r11);
                _sum0 = __msa_fmadd_w(_sum0, _k12, _r12);
                _sum0 = __msa_fmadd_w(_sum0, _k13, _r13);
                _sum0 = __msa_fmadd_w(_sum0, _k14, _r14);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);
                v4f32 _r23 = (v4f32)__msa_ld_w(r2 + 4 * 3, 0);
                v4f32 _r24 = (v4f32)__msa_ld_w(r2 + 4 * 4, 0);

                _sum1 = __msa_fmadd_w(_sum1, _k10, _r20);
                _sum1 = __msa_fmadd_w(_sum1, _k11, _r21);
                _sum1 = __msa_fmadd_w(_sum1, _k12, _r22);
                _sum1 = __msa_fmadd_w(_sum1, _k13, _r23);
                _sum1 = __msa_fmadd_w(_sum1, _k14, _r24);

                v4f32 _k20 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k21 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k22 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k23 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k24 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k20, _r20);
                _sum0 = __msa_fmadd_w(_sum0, _k21, _r21);
                _sum0 = __msa_fmadd_w(_sum0, _k22, _r22);
                _sum0 = __msa_fmadd_w(_sum0, _k23, _r23);
                _sum0 = __msa_fmadd_w(_sum0, _k24, _r24);

                v4f32 _r30 = (v4f32)__msa_ld_w(r3, 0);
                v4f32 _r31 = (v4f32)__msa_ld_w(r3 + 4, 0);
                v4f32 _r32 = (v4f32)__msa_ld_w(r3 + 4 * 2, 0);
                v4f32 _r33 = (v4f32)__msa_ld_w(r3 + 4 * 3, 0);
                v4f32 _r34 = (v4f32)__msa_ld_w(r3 + 4 * 4, 0);

                _sum1 = __msa_fmadd_w(_sum1, _k20, _r30);
                _sum1 = __msa_fmadd_w(_sum1, _k21, _r31);
                _sum1 = __msa_fmadd_w(_sum1, _k22, _r32);
                _sum1 = __msa_fmadd_w(_sum1, _k23, _r33);
                _sum1 = __msa_fmadd_w(_sum1, _k24, _r34);

                v4f32 _k30 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k31 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k32 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k33 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k34 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k30, _r30);
                _sum0 = __msa_fmadd_w(_sum0, _k31, _r31);
                _sum0 = __msa_fmadd_w(_sum0, _k32, _r32);
                _sum0 = __msa_fmadd_w(_sum0, _k33, _r33);
                _sum0 = __msa_fmadd_w(_sum0, _k34, _r34);

                v4f32 _r40 = (v4f32)__msa_ld_w(r4, 0);
                v4f32 _r41 = (v4f32)__msa_ld_w(r4 + 4, 0);
                v4f32 _r42 = (v4f32)__msa_ld_w(r4 + 4 * 2, 0);
                v4f32 _r43 = (v4f32)__msa_ld_w(r4 + 4 * 3, 0);
                v4f32 _r44 = (v4f32)__msa_ld_w(r4 + 4 * 4, 0);

                _sum1 = __msa_fmadd_w(_sum1, _k30, _r40);
                _sum1 = __msa_fmadd_w(_sum1, _k31, _r41);
                _sum1 = __msa_fmadd_w(_sum1, _k32, _r42);
                _sum1 = __msa_fmadd_w(_sum1, _k33, _r43);
                _sum1 = __msa_fmadd_w(_sum1, _k34, _r44);

                v4f32 _k40 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k41 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k42 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k43 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k44 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 -= 4 * 20;

                _sum0 = __msa_fmadd_w(_sum0, _k40, _r40);
                _sum0 = __msa_fmadd_w(_sum0, _k41, _r41);
                _sum0 = __msa_fmadd_w(_sum0, _k42, _r42);
                _sum0 = __msa_fmadd_w(_sum0, _k43, _r43);
                _sum0 = __msa_fmadd_w(_sum0, _k44, _r44);

                v4f32 _r50 = (v4f32)__msa_ld_w(r5, 0);
                v4f32 _r51 = (v4f32)__msa_ld_w(r5 + 4, 0);
                v4f32 _r52 = (v4f32)__msa_ld_w(r5 + 4 * 2, 0);
                v4f32 _r53 = (v4f32)__msa_ld_w(r5 + 4 * 3, 0);
                v4f32 _r54 = (v4f32)__msa_ld_w(r5 + 4 * 4, 0);

                _sum1 = __msa_fmadd_w(_sum1, _k40, _r50);
                _sum1 = __msa_fmadd_w(_sum1, _k41, _r51);
                _sum1 = __msa_fmadd_w(_sum1, _k42, _r52);
                _sum1 = __msa_fmadd_w(_sum1, _k43, _r53);
                _sum1 = __msa_fmadd_w(_sum1, _k44, _r54);

                __msa_st_w((v4i32)_sum0, outptr0, 0);
                __msa_st_w((v4i32)_sum1, outptr1, 0);

                outptr0 += 4;
                outptr1 += 4;

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
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
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j < outw; j++)
            {
                __builtin_prefetch(r0 + 16);
                __builtin_prefetch(r1 + 16);
                __builtin_prefetch(r2 + 16);
                __builtin_prefetch(r3 + 16);
                __builtin_prefetch(r4 + 16);

                __builtin_prefetch(k0 + 400);

                v4f32 _sum0 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);
                v4f32 _r04 = (v4f32)__msa_ld_w(r0 + 4 * 4, 0);

                v4f32 _k00 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k01 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k02 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k03 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k04 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k00, _r00);
                _sum0 = __msa_fmadd_w(_sum0, _k01, _r01);
                _sum0 = __msa_fmadd_w(_sum0, _k02, _r02);
                _sum0 = __msa_fmadd_w(_sum0, _k03, _r03);
                _sum0 = __msa_fmadd_w(_sum0, _k04, _r04);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);
                v4f32 _r13 = (v4f32)__msa_ld_w(r1 + 4 * 3, 0);
                v4f32 _r14 = (v4f32)__msa_ld_w(r1 + 4 * 4, 0);

                v4f32 _k10 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k11 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k12 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k13 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k14 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k10, _r10);
                _sum0 = __msa_fmadd_w(_sum0, _k11, _r11);
                _sum0 = __msa_fmadd_w(_sum0, _k12, _r12);
                _sum0 = __msa_fmadd_w(_sum0, _k13, _r13);
                _sum0 = __msa_fmadd_w(_sum0, _k14, _r14);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);
                v4f32 _r23 = (v4f32)__msa_ld_w(r2 + 4 * 3, 0);
                v4f32 _r24 = (v4f32)__msa_ld_w(r2 + 4 * 4, 0);

                v4f32 _k20 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k21 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k22 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k23 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k24 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k20, _r20);
                _sum0 = __msa_fmadd_w(_sum0, _k21, _r21);
                _sum0 = __msa_fmadd_w(_sum0, _k22, _r22);
                _sum0 = __msa_fmadd_w(_sum0, _k23, _r23);
                _sum0 = __msa_fmadd_w(_sum0, _k24, _r24);

                v4f32 _r30 = (v4f32)__msa_ld_w(r3, 0);
                v4f32 _r31 = (v4f32)__msa_ld_w(r3 + 4, 0);
                v4f32 _r32 = (v4f32)__msa_ld_w(r3 + 4 * 2, 0);
                v4f32 _r33 = (v4f32)__msa_ld_w(r3 + 4 * 3, 0);
                v4f32 _r34 = (v4f32)__msa_ld_w(r3 + 4 * 4, 0);

                v4f32 _k30 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k31 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k32 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k33 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k34 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k30, _r30);
                _sum0 = __msa_fmadd_w(_sum0, _k31, _r31);
                _sum0 = __msa_fmadd_w(_sum0, _k32, _r32);
                _sum0 = __msa_fmadd_w(_sum0, _k33, _r33);
                _sum0 = __msa_fmadd_w(_sum0, _k34, _r34);

                v4f32 _r40 = (v4f32)__msa_ld_w(r4, 0);
                v4f32 _r41 = (v4f32)__msa_ld_w(r4 + 4, 0);
                v4f32 _r42 = (v4f32)__msa_ld_w(r4 + 4 * 2, 0);
                v4f32 _r43 = (v4f32)__msa_ld_w(r4 + 4 * 3, 0);
                v4f32 _r44 = (v4f32)__msa_ld_w(r4 + 4 * 4, 0);

                v4f32 _k40 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k41 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k42 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k43 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k44 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 -= 4 * 20;

                _sum0 = __msa_fmadd_w(_sum0, _k40, _r40);
                _sum0 = __msa_fmadd_w(_sum0, _k41, _r41);
                _sum0 = __msa_fmadd_w(_sum0, _k42, _r42);
                _sum0 = __msa_fmadd_w(_sum0, _k43, _r43);
                _sum0 = __msa_fmadd_w(_sum0, _k44, _r44);

                __msa_st_w((v4i32)_sum0, outptr0, 0);

                outptr0 += 4;

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
            }

            r0 += 4 * 4;
            r1 += 4 * 4;
            r2 += 4 * 4;
            r3 += 4 * 4;
            r4 += 4 * 4;
        }
    }
}

static void convdw5x5s2_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        v4f32 _bias0 = bias ? (v4f32)__msa_ld_w(bias + g * 4, 0) : (v4f32)__msa_fill_w(0);

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
            for (; j < outw; j++)
            {
                __builtin_prefetch(r0 + 32);
                __builtin_prefetch(r1 + 32);
                __builtin_prefetch(r2 + 32);
                __builtin_prefetch(r3 + 32);
                __builtin_prefetch(r4 + 32);

                __builtin_prefetch(k0 + 400);

                v4f32 _sum0 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);
                v4f32 _r04 = (v4f32)__msa_ld_w(r0 + 4 * 4, 0);

                v4f32 _k00 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k01 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k02 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k03 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k04 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k00, _r00);
                _sum0 = __msa_fmadd_w(_sum0, _k01, _r01);
                _sum0 = __msa_fmadd_w(_sum0, _k02, _r02);
                _sum0 = __msa_fmadd_w(_sum0, _k03, _r03);
                _sum0 = __msa_fmadd_w(_sum0, _k04, _r04);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);
                v4f32 _r13 = (v4f32)__msa_ld_w(r1 + 4 * 3, 0);
                v4f32 _r14 = (v4f32)__msa_ld_w(r1 + 4 * 4, 0);

                v4f32 _k10 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k11 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k12 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k13 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k14 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k10, _r10);
                _sum0 = __msa_fmadd_w(_sum0, _k11, _r11);
                _sum0 = __msa_fmadd_w(_sum0, _k12, _r12);
                _sum0 = __msa_fmadd_w(_sum0, _k13, _r13);
                _sum0 = __msa_fmadd_w(_sum0, _k14, _r14);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);
                v4f32 _r23 = (v4f32)__msa_ld_w(r2 + 4 * 3, 0);
                v4f32 _r24 = (v4f32)__msa_ld_w(r2 + 4 * 4, 0);

                v4f32 _k20 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k21 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k22 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k23 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k24 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k20, _r20);
                _sum0 = __msa_fmadd_w(_sum0, _k21, _r21);
                _sum0 = __msa_fmadd_w(_sum0, _k22, _r22);
                _sum0 = __msa_fmadd_w(_sum0, _k23, _r23);
                _sum0 = __msa_fmadd_w(_sum0, _k24, _r24);

                v4f32 _r30 = (v4f32)__msa_ld_w(r3, 0);
                v4f32 _r31 = (v4f32)__msa_ld_w(r3 + 4, 0);
                v4f32 _r32 = (v4f32)__msa_ld_w(r3 + 4 * 2, 0);
                v4f32 _r33 = (v4f32)__msa_ld_w(r3 + 4 * 3, 0);
                v4f32 _r34 = (v4f32)__msa_ld_w(r3 + 4 * 4, 0);

                v4f32 _k30 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k31 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k32 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k33 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k34 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __msa_fmadd_w(_sum0, _k30, _r30);
                _sum0 = __msa_fmadd_w(_sum0, _k31, _r31);
                _sum0 = __msa_fmadd_w(_sum0, _k32, _r32);
                _sum0 = __msa_fmadd_w(_sum0, _k33, _r33);
                _sum0 = __msa_fmadd_w(_sum0, _k34, _r34);

                v4f32 _r40 = (v4f32)__msa_ld_w(r4, 0);
                v4f32 _r41 = (v4f32)__msa_ld_w(r4 + 4, 0);
                v4f32 _r42 = (v4f32)__msa_ld_w(r4 + 4 * 2, 0);
                v4f32 _r43 = (v4f32)__msa_ld_w(r4 + 4 * 3, 0);
                v4f32 _r44 = (v4f32)__msa_ld_w(r4 + 4 * 4, 0);

                v4f32 _k40 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _k41 = (v4f32)__msa_ld_w(k0 + 4, 0);
                v4f32 _k42 = (v4f32)__msa_ld_w(k0 + 4 * 2, 0);
                v4f32 _k43 = (v4f32)__msa_ld_w(k0 + 4 * 3, 0);
                v4f32 _k44 = (v4f32)__msa_ld_w(k0 + 4 * 4, 0);
                k0 -= 4 * 20;

                _sum0 = __msa_fmadd_w(_sum0, _k40, _r40);
                _sum0 = __msa_fmadd_w(_sum0, _k41, _r41);
                _sum0 = __msa_fmadd_w(_sum0, _k42, _r42);
                _sum0 = __msa_fmadd_w(_sum0, _k43, _r43);
                _sum0 = __msa_fmadd_w(_sum0, _k44, _r44);

                __msa_st_w((v4i32)_sum0, outptr0, 0);

                outptr0 += 4;

                r0 += 4 * 2;
                r1 += 4 * 2;
                r2 += 4 * 2;
                r3 += 4 * 2;
                r4 += 4 * 2;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }
}

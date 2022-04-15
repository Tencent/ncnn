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

static void convdw3x3s1_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
                __builtin_prefetch(r0 + 32);
                __builtin_prefetch(r1 + 32);
                __builtin_prefetch(r2 + 32);
                __builtin_prefetch(r3 + 32);

                v4f32 _sum00 = _bias0;
                v4f32 _sum01 = _bias0;
                v4f32 _sum10 = _bias0;
                v4f32 _sum11 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k00, _r00);
                _sum00 = __msa_fmadd_w(_sum00, _k01, _r01);
                _sum00 = __msa_fmadd_w(_sum00, _k02, _r02);
                _sum01 = __msa_fmadd_w(_sum01, _k00, _r01);
                _sum01 = __msa_fmadd_w(_sum01, _k01, _r02);
                _sum01 = __msa_fmadd_w(_sum01, _k02, _r03);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);
                v4f32 _r13 = (v4f32)__msa_ld_w(r1 + 4 * 3, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k10, _r10);
                _sum00 = __msa_fmadd_w(_sum00, _k11, _r11);
                _sum00 = __msa_fmadd_w(_sum00, _k12, _r12);
                _sum01 = __msa_fmadd_w(_sum01, _k10, _r11);
                _sum01 = __msa_fmadd_w(_sum01, _k11, _r12);
                _sum01 = __msa_fmadd_w(_sum01, _k12, _r13);
                _sum10 = __msa_fmadd_w(_sum10, _k00, _r10);
                _sum10 = __msa_fmadd_w(_sum10, _k01, _r11);
                _sum10 = __msa_fmadd_w(_sum10, _k02, _r12);
                _sum11 = __msa_fmadd_w(_sum11, _k00, _r11);
                _sum11 = __msa_fmadd_w(_sum11, _k01, _r12);
                _sum11 = __msa_fmadd_w(_sum11, _k02, _r13);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);
                v4f32 _r23 = (v4f32)__msa_ld_w(r2 + 4 * 3, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k20, _r20);
                _sum00 = __msa_fmadd_w(_sum00, _k21, _r21);
                _sum00 = __msa_fmadd_w(_sum00, _k22, _r22);
                _sum01 = __msa_fmadd_w(_sum01, _k20, _r21);
                _sum01 = __msa_fmadd_w(_sum01, _k21, _r22);
                _sum01 = __msa_fmadd_w(_sum01, _k22, _r23);
                _sum10 = __msa_fmadd_w(_sum10, _k10, _r20);
                _sum10 = __msa_fmadd_w(_sum10, _k11, _r21);
                _sum10 = __msa_fmadd_w(_sum10, _k12, _r22);
                _sum11 = __msa_fmadd_w(_sum11, _k10, _r21);
                _sum11 = __msa_fmadd_w(_sum11, _k11, _r22);
                _sum11 = __msa_fmadd_w(_sum11, _k12, _r23);

                v4f32 _r30 = (v4f32)__msa_ld_w(r3, 0);
                v4f32 _r31 = (v4f32)__msa_ld_w(r3 + 4, 0);
                v4f32 _r32 = (v4f32)__msa_ld_w(r3 + 4 * 2, 0);
                v4f32 _r33 = (v4f32)__msa_ld_w(r3 + 4 * 3, 0);

                _sum10 = __msa_fmadd_w(_sum10, _k20, _r30);
                _sum10 = __msa_fmadd_w(_sum10, _k21, _r31);
                _sum10 = __msa_fmadd_w(_sum10, _k22, _r32);
                _sum11 = __msa_fmadd_w(_sum11, _k20, _r31);
                _sum11 = __msa_fmadd_w(_sum11, _k21, _r32);
                _sum11 = __msa_fmadd_w(_sum11, _k22, _r33);

                __msa_st_w((v4i32)_sum00, outptr0, 0);
                __msa_st_w((v4i32)_sum01, outptr0 + 4, 0);
                __msa_st_w((v4i32)_sum10, outptr1, 0);
                __msa_st_w((v4i32)_sum11, outptr1 + 4, 0);

                outptr0 += 4 * 2;
                outptr1 += 4 * 2;

                r0 += 4 * 2;
                r1 += 4 * 2;
                r2 += 4 * 2;
                r3 += 4 * 2;
            }
            for (; j < outw; j++)
            {
                __builtin_prefetch(r0 + 16);
                __builtin_prefetch(r1 + 16);
                __builtin_prefetch(r2 + 16);
                __builtin_prefetch(r3 + 16);

                v4f32 _sum0 = _bias0;
                v4f32 _sum1 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k00, _r00);
                _sum0 = __msa_fmadd_w(_sum0, _k01, _r01);
                _sum0 = __msa_fmadd_w(_sum0, _k02, _r02);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k10, _r10);
                _sum0 = __msa_fmadd_w(_sum0, _k11, _r11);
                _sum0 = __msa_fmadd_w(_sum0, _k12, _r12);
                _sum1 = __msa_fmadd_w(_sum1, _k00, _r10);
                _sum1 = __msa_fmadd_w(_sum1, _k01, _r11);
                _sum1 = __msa_fmadd_w(_sum1, _k02, _r12);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k20, _r20);
                _sum0 = __msa_fmadd_w(_sum0, _k21, _r21);
                _sum0 = __msa_fmadd_w(_sum0, _k22, _r22);
                _sum1 = __msa_fmadd_w(_sum1, _k10, _r20);
                _sum1 = __msa_fmadd_w(_sum1, _k11, _r21);
                _sum1 = __msa_fmadd_w(_sum1, _k12, _r22);

                v4f32 _r30 = (v4f32)__msa_ld_w(r3, 0);
                v4f32 _r31 = (v4f32)__msa_ld_w(r3 + 4, 0);
                v4f32 _r32 = (v4f32)__msa_ld_w(r3 + 4 * 2, 0);

                _sum1 = __msa_fmadd_w(_sum1, _k20, _r30);
                _sum1 = __msa_fmadd_w(_sum1, _k21, _r31);
                _sum1 = __msa_fmadd_w(_sum1, _k22, _r32);

                __msa_st_w((v4i32)_sum0, outptr0, 0);
                __msa_st_w((v4i32)_sum1, outptr1, 0);

                outptr0 += 4;
                outptr1 += 4;

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
            }

            r0 += 2 * 4 + w * 4;
            r1 += 2 * 4 + w * 4;
            r2 += 2 * 4 + w * 4;
            r3 += 2 * 4 + w * 4;

            outptr0 += outw * 4;
            outptr1 += outw * 4;
        }
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
                __builtin_prefetch(r0 + 32);
                __builtin_prefetch(r1 + 32);
                __builtin_prefetch(r2 + 32);

                v4f32 _sum00 = _bias0;
                v4f32 _sum01 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k00, _r00);
                _sum00 = __msa_fmadd_w(_sum00, _k01, _r01);
                _sum00 = __msa_fmadd_w(_sum00, _k02, _r02);
                _sum01 = __msa_fmadd_w(_sum01, _k00, _r01);
                _sum01 = __msa_fmadd_w(_sum01, _k01, _r02);
                _sum01 = __msa_fmadd_w(_sum01, _k02, _r03);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);
                v4f32 _r13 = (v4f32)__msa_ld_w(r1 + 4 * 3, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k10, _r10);
                _sum00 = __msa_fmadd_w(_sum00, _k11, _r11);
                _sum00 = __msa_fmadd_w(_sum00, _k12, _r12);
                _sum01 = __msa_fmadd_w(_sum01, _k10, _r11);
                _sum01 = __msa_fmadd_w(_sum01, _k11, _r12);
                _sum01 = __msa_fmadd_w(_sum01, _k12, _r13);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);
                v4f32 _r23 = (v4f32)__msa_ld_w(r2 + 4 * 3, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k20, _r20);
                _sum00 = __msa_fmadd_w(_sum00, _k21, _r21);
                _sum00 = __msa_fmadd_w(_sum00, _k22, _r22);
                _sum01 = __msa_fmadd_w(_sum01, _k20, _r21);
                _sum01 = __msa_fmadd_w(_sum01, _k21, _r22);
                _sum01 = __msa_fmadd_w(_sum01, _k22, _r23);

                __msa_st_w((v4i32)_sum00, outptr0, 0);
                __msa_st_w((v4i32)_sum01, outptr0 + 4, 0);

                outptr0 += 4 * 2;

                r0 += 4 * 2;
                r1 += 4 * 2;
                r2 += 4 * 2;
            }
            for (; j < outw; j++)
            {
                __builtin_prefetch(r0 + 16);
                __builtin_prefetch(r1 + 16);
                __builtin_prefetch(r2 + 16);

                v4f32 _sum0 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k00, _r00);
                _sum0 = __msa_fmadd_w(_sum0, _k01, _r01);
                _sum0 = __msa_fmadd_w(_sum0, _k02, _r02);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k10, _r10);
                _sum0 = __msa_fmadd_w(_sum0, _k11, _r11);
                _sum0 = __msa_fmadd_w(_sum0, _k12, _r12);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k20, _r20);
                _sum0 = __msa_fmadd_w(_sum0, _k21, _r21);
                _sum0 = __msa_fmadd_w(_sum0, _k22, _r22);

                __msa_st_w((v4i32)_sum0, outptr0, 0);

                outptr0 += 4;

                r0 += 4;
                r1 += 4;
                r2 += 4;
            }

            r0 += 2 * 4;
            r1 += 2 * 4;
            r2 += 2 * 4;
        }
    }
}

static void convdw3x3s2_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
            for (; j + 1 < outw; j += 2)
            {
                __builtin_prefetch(r0 + 64);
                __builtin_prefetch(r1 + 64);
                __builtin_prefetch(r2 + 64);

                v4f32 _sum00 = _bias0;
                v4f32 _sum01 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);
                v4f32 _r04 = (v4f32)__msa_ld_w(r0 + 4 * 4, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k00, _r00);
                _sum00 = __msa_fmadd_w(_sum00, _k01, _r01);
                _sum00 = __msa_fmadd_w(_sum00, _k02, _r02);
                _sum01 = __msa_fmadd_w(_sum01, _k00, _r02);
                _sum01 = __msa_fmadd_w(_sum01, _k01, _r03);
                _sum01 = __msa_fmadd_w(_sum01, _k02, _r04);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);
                v4f32 _r13 = (v4f32)__msa_ld_w(r1 + 4 * 3, 0);
                v4f32 _r14 = (v4f32)__msa_ld_w(r1 + 4 * 4, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k10, _r10);
                _sum00 = __msa_fmadd_w(_sum00, _k11, _r11);
                _sum00 = __msa_fmadd_w(_sum00, _k12, _r12);
                _sum01 = __msa_fmadd_w(_sum01, _k10, _r12);
                _sum01 = __msa_fmadd_w(_sum01, _k11, _r13);
                _sum01 = __msa_fmadd_w(_sum01, _k12, _r14);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);
                v4f32 _r23 = (v4f32)__msa_ld_w(r2 + 4 * 3, 0);
                v4f32 _r24 = (v4f32)__msa_ld_w(r2 + 4 * 4, 0);

                _sum00 = __msa_fmadd_w(_sum00, _k20, _r20);
                _sum00 = __msa_fmadd_w(_sum00, _k21, _r21);
                _sum00 = __msa_fmadd_w(_sum00, _k22, _r22);
                _sum01 = __msa_fmadd_w(_sum01, _k20, _r22);
                _sum01 = __msa_fmadd_w(_sum01, _k21, _r23);
                _sum01 = __msa_fmadd_w(_sum01, _k22, _r24);

                __msa_st_w((v4i32)_sum00, outptr0, 0);
                __msa_st_w((v4i32)_sum01, outptr0 + 4, 0);

                outptr0 += 4 * 2;

                r0 += 4 * 4;
                r1 += 4 * 4;
                r2 += 4 * 4;
            }
            for (; j < outw; j++)
            {
                __builtin_prefetch(r0 + 32);
                __builtin_prefetch(r1 + 32);
                __builtin_prefetch(r2 + 32);

                v4f32 _sum0 = _bias0;

                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k00, _r00);
                _sum0 = __msa_fmadd_w(_sum0, _k01, _r01);
                _sum0 = __msa_fmadd_w(_sum0, _k02, _r02);

                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k10, _r10);
                _sum0 = __msa_fmadd_w(_sum0, _k11, _r11);
                _sum0 = __msa_fmadd_w(_sum0, _k12, _r12);

                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 4 * 2, 0);

                _sum0 = __msa_fmadd_w(_sum0, _k20, _r20);
                _sum0 = __msa_fmadd_w(_sum0, _k21, _r21);
                _sum0 = __msa_fmadd_w(_sum0, _k22, _r22);

                __msa_st_w((v4i32)_sum0, outptr0, 0);

                outptr0 += 4;

                r0 += 4 * 2;
                r1 += 4 * 2;
                r2 += 4 * 2;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

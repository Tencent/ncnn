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

static void convdw3x3s1_pack4_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        __m128 _bias0 = bias ? (__m128)__lsx_vld(bias + g * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);
        float* outptr1 = out.row(1);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        const float* r3 = img0.row(3);

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
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
                __builtin_prefetch(r0 + 32);
                __builtin_prefetch(r1 + 32);
                __builtin_prefetch(r2 + 32);
                __builtin_prefetch(r3 + 32);

                __m128 _sum00 = _bias0;
                __m128 _sum01 = _bias0;
                __m128 _sum10 = _bias0;
                __m128 _sum11 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);

                _sum00 = __lsx_vfmadd_s(_r00, _k00, _sum00);
                _sum00 = __lsx_vfmadd_s(_r01, _k01, _sum00);
                _sum00 = __lsx_vfmadd_s(_r02, _k02, _sum00);
                _sum01 = __lsx_vfmadd_s(_r01, _k00, _sum01);
                _sum01 = __lsx_vfmadd_s(_r02, _k01, _sum01);
                _sum01 = __lsx_vfmadd_s(_r03, _k02, _sum01);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);
                __m128 _r13 = (__m128)__lsx_vld(r1 + 4 * 3, 0);

                _sum00 = __lsx_vfmadd_s(_r10, _k10, _sum00);
                _sum00 = __lsx_vfmadd_s(_r11, _k11, _sum00);
                _sum00 = __lsx_vfmadd_s(_r12, _k12, _sum00);
                _sum01 = __lsx_vfmadd_s(_r11, _k10, _sum01);
                _sum01 = __lsx_vfmadd_s(_r12, _k11, _sum01);
                _sum01 = __lsx_vfmadd_s(_r13, _k12, _sum01);
                _sum10 = __lsx_vfmadd_s(_r10, _k00, _sum10);
                _sum10 = __lsx_vfmadd_s(_r11, _k01, _sum10);
                _sum10 = __lsx_vfmadd_s(_r12, _k02, _sum10);
                _sum11 = __lsx_vfmadd_s(_r11, _k00, _sum11);
                _sum11 = __lsx_vfmadd_s(_r12, _k01, _sum11);
                _sum11 = __lsx_vfmadd_s(_r13, _k02, _sum11);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);
                __m128 _r23 = (__m128)__lsx_vld(r2 + 4 * 3, 0);

                _sum00 = __lsx_vfmadd_s(_r20, _k20, _sum00);
                _sum00 = __lsx_vfmadd_s(_r21, _k21, _sum00);
                _sum00 = __lsx_vfmadd_s(_r22, _k22, _sum00);
                _sum01 = __lsx_vfmadd_s(_r21, _k20, _sum01);
                _sum01 = __lsx_vfmadd_s(_r22, _k21, _sum01);
                _sum01 = __lsx_vfmadd_s(_r23, _k22, _sum01);
                _sum10 = __lsx_vfmadd_s(_r20, _k10, _sum10);
                _sum10 = __lsx_vfmadd_s(_r21, _k11, _sum10);
                _sum10 = __lsx_vfmadd_s(_r22, _k12, _sum10);
                _sum11 = __lsx_vfmadd_s(_r21, _k10, _sum11);
                _sum11 = __lsx_vfmadd_s(_r22, _k11, _sum11);
                _sum11 = __lsx_vfmadd_s(_r23, _k12, _sum11);

                __m128 _r30 = (__m128)__lsx_vld(r3, 0);
                __m128 _r31 = (__m128)__lsx_vld(r3 + 4, 0);
                __m128 _r32 = (__m128)__lsx_vld(r3 + 4 * 2, 0);
                __m128 _r33 = (__m128)__lsx_vld(r3 + 4 * 3, 0);

                _sum10 = __lsx_vfmadd_s(_r30, _k20, _sum10);
                _sum10 = __lsx_vfmadd_s(_r31, _k21, _sum10);
                _sum10 = __lsx_vfmadd_s(_r32, _k22, _sum10);
                _sum11 = __lsx_vfmadd_s(_r31, _k20, _sum11);
                _sum11 = __lsx_vfmadd_s(_r32, _k21, _sum11);
                _sum11 = __lsx_vfmadd_s(_r33, _k22, _sum11);

                __lsx_vst(_sum00, outptr0, 0);
                __lsx_vst(_sum01, outptr0 + 4, 0);
                __lsx_vst(_sum10, outptr1, 0);
                __lsx_vst(_sum11, outptr1 + 4, 0);

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

                __m128 _sum0 = _bias0;
                __m128 _sum1 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r00, _k00, _sum0);
                _sum0 = __lsx_vfmadd_s(_r01, _k01, _sum0);
                _sum0 = __lsx_vfmadd_s(_r02, _k02, _sum0);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r10, _k10, _sum0);
                _sum0 = __lsx_vfmadd_s(_r11, _k11, _sum0);
                _sum0 = __lsx_vfmadd_s(_r12, _k12, _sum0);
                _sum1 = __lsx_vfmadd_s(_r10, _k00, _sum1);
                _sum1 = __lsx_vfmadd_s(_r11, _k01, _sum1);
                _sum1 = __lsx_vfmadd_s(_r12, _k02, _sum1);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r20, _k20, _sum0);
                _sum0 = __lsx_vfmadd_s(_r21, _k21, _sum0);
                _sum0 = __lsx_vfmadd_s(_r22, _k22, _sum0);
                _sum1 = __lsx_vfmadd_s(_r20, _k10, _sum1);
                _sum1 = __lsx_vfmadd_s(_r21, _k11, _sum1);
                _sum1 = __lsx_vfmadd_s(_r22, _k12, _sum1);

                __m128 _r30 = (__m128)__lsx_vld(r3, 0);
                __m128 _r31 = (__m128)__lsx_vld(r3 + 4, 0);
                __m128 _r32 = (__m128)__lsx_vld(r3 + 4 * 2, 0);

                _sum1 = __lsx_vfmadd_s(_r30, _k20, _sum1);
                _sum1 = __lsx_vfmadd_s(_r31, _k21, _sum1);
                _sum1 = __lsx_vfmadd_s(_r32, _k22, _sum1);

                __lsx_vst(_sum0, outptr0, 0);
                __lsx_vst(_sum1, outptr1, 0);

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

                __m128 _sum00 = _bias0;
                __m128 _sum01 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);

                _sum00 = __lsx_vfmadd_s(_r00, _k00, _sum00);
                _sum00 = __lsx_vfmadd_s(_r01, _k01, _sum00);
                _sum00 = __lsx_vfmadd_s(_r02, _k02, _sum00);
                _sum01 = __lsx_vfmadd_s(_r01, _k00, _sum01);
                _sum01 = __lsx_vfmadd_s(_r02, _k01, _sum01);
                _sum01 = __lsx_vfmadd_s(_r03, _k02, _sum01);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);
                __m128 _r13 = (__m128)__lsx_vld(r1 + 4 * 3, 0);

                _sum00 = __lsx_vfmadd_s(_r10, _k10, _sum00);
                _sum00 = __lsx_vfmadd_s(_r11, _k11, _sum00);
                _sum00 = __lsx_vfmadd_s(_r12, _k12, _sum00);
                _sum01 = __lsx_vfmadd_s(_r11, _k10, _sum01);
                _sum01 = __lsx_vfmadd_s(_r12, _k11, _sum01);
                _sum01 = __lsx_vfmadd_s(_r13, _k12, _sum01);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);
                __m128 _r23 = (__m128)__lsx_vld(r2 + 4 * 3, 0);

                _sum00 = __lsx_vfmadd_s(_r20, _k20, _sum00);
                _sum00 = __lsx_vfmadd_s(_r21, _k21, _sum00);
                _sum00 = __lsx_vfmadd_s(_r22, _k22, _sum00);
                _sum01 = __lsx_vfmadd_s(_r21, _k20, _sum01);
                _sum01 = __lsx_vfmadd_s(_r22, _k21, _sum01);
                _sum01 = __lsx_vfmadd_s(_r23, _k22, _sum01);

                __lsx_vst(_sum00, outptr0, 0);
                __lsx_vst(_sum01, outptr0 + 4, 0);

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

                __m128 _sum0 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r00, _k00, _sum0);
                _sum0 = __lsx_vfmadd_s(_r01, _k01, _sum0);
                _sum0 = __lsx_vfmadd_s(_r02, _k02, _sum0);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r10, _k10, _sum0);
                _sum0 = __lsx_vfmadd_s(_r11, _k11, _sum0);
                _sum0 = __lsx_vfmadd_s(_r12, _k12, _sum0);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r20, _k20, _sum0);
                _sum0 = __lsx_vfmadd_s(_r21, _k21, _sum0);
                _sum0 = __lsx_vfmadd_s(_r22, _k22, _sum0);

                __lsx_vst(_sum0, outptr0, 0);

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

static void convdw3x3s2_pack4_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        __m128 _bias0 = bias ? (__m128)__lsx_vld(bias + g * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);

        const float* k0 = kernel.row(g);

        float* outptr0 = out;

        const Mat img0 = bottom_blob.channel(g);

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
            for (; j + 1 < outw; j += 2)
            {
                __builtin_prefetch(r0 + 64);
                __builtin_prefetch(r1 + 64);
                __builtin_prefetch(r2 + 64);

                __m128 _sum00 = _bias0;
                __m128 _sum01 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);
                __m128 _r04 = (__m128)__lsx_vld(r0 + 4 * 4, 0);

                _sum00 = __lsx_vfmadd_s(_r00, _k00, _sum00);
                _sum00 = __lsx_vfmadd_s(_r01, _k01, _sum00);
                _sum00 = __lsx_vfmadd_s(_r02, _k02, _sum00);
                _sum01 = __lsx_vfmadd_s(_r02, _k00, _sum01);
                _sum01 = __lsx_vfmadd_s(_r03, _k01, _sum01);
                _sum01 = __lsx_vfmadd_s(_r04, _k02, _sum01);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);
                __m128 _r13 = (__m128)__lsx_vld(r1 + 4 * 3, 0);
                __m128 _r14 = (__m128)__lsx_vld(r1 + 4 * 4, 0);

                _sum00 = __lsx_vfmadd_s(_r10, _k10, _sum00);
                _sum00 = __lsx_vfmadd_s(_r11, _k11, _sum00);
                _sum00 = __lsx_vfmadd_s(_r12, _k12, _sum00);
                _sum01 = __lsx_vfmadd_s(_r12, _k10, _sum01);
                _sum01 = __lsx_vfmadd_s(_r13, _k11, _sum01);
                _sum01 = __lsx_vfmadd_s(_r14, _k12, _sum01);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);
                __m128 _r23 = (__m128)__lsx_vld(r2 + 4 * 3, 0);
                __m128 _r24 = (__m128)__lsx_vld(r2 + 4 * 4, 0);

                _sum00 = __lsx_vfmadd_s(_r20, _k20, _sum00);
                _sum00 = __lsx_vfmadd_s(_r21, _k21, _sum00);
                _sum00 = __lsx_vfmadd_s(_r22, _k22, _sum00);
                _sum01 = __lsx_vfmadd_s(_r22, _k20, _sum01);
                _sum01 = __lsx_vfmadd_s(_r23, _k21, _sum01);
                _sum01 = __lsx_vfmadd_s(_r24, _k22, _sum01);

                __lsx_vst(_sum00, outptr0, 0);
                __lsx_vst(_sum01, outptr0 + 4, 0);

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

                __m128 _sum0 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r00, _k00, _sum0);
                _sum0 = __lsx_vfmadd_s(_r01, _k01, _sum0);
                _sum0 = __lsx_vfmadd_s(_r02, _k02, _sum0);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r10, _k10, _sum0);
                _sum0 = __lsx_vfmadd_s(_r11, _k11, _sum0);
                _sum0 = __lsx_vfmadd_s(_r12, _k12, _sum0);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);

                _sum0 = __lsx_vfmadd_s(_r20, _k20, _sum0);
                _sum0 = __lsx_vfmadd_s(_r21, _k21, _sum0);
                _sum0 = __lsx_vfmadd_s(_r22, _k22, _sum0);

                __lsx_vst(_sum0, outptr0, 0);

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

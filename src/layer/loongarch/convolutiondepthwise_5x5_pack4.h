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

static void convdw5x5s1_pack4_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

                __m128 _sum0 = _bias0;
                __m128 _sum1 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);
                __m128 _r04 = (__m128)__lsx_vld(r0 + 4 * 4, 0);

                __m128 _k00 = (__m128)__lsx_vld(k0, 0);
                __m128 _k01 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k02 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k03 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k04 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r00, _k00, _sum0);
                _sum0 = __lsx_vfmadd_s(_r01, _k01, _sum0);
                _sum0 = __lsx_vfmadd_s(_r02, _k02, _sum0);
                _sum0 = __lsx_vfmadd_s(_r03, _k03, _sum0);
                _sum0 = __lsx_vfmadd_s(_r04, _k04, _sum0);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);
                __m128 _r13 = (__m128)__lsx_vld(r1 + 4 * 3, 0);
                __m128 _r14 = (__m128)__lsx_vld(r1 + 4 * 4, 0);

                _sum1 = __lsx_vfmadd_s(_r10, _k00, _sum1);
                _sum1 = __lsx_vfmadd_s(_r11, _k01, _sum1);
                _sum1 = __lsx_vfmadd_s(_r12, _k02, _sum1);
                _sum1 = __lsx_vfmadd_s(_r13, _k03, _sum1);
                _sum1 = __lsx_vfmadd_s(_r14, _k04, _sum1);

                __m128 _k10 = (__m128)__lsx_vld(k0, 0);
                __m128 _k11 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k12 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k13 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k14 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r10, _k10, _sum0);
                _sum0 = __lsx_vfmadd_s(_r11, _k11, _sum0);
                _sum0 = __lsx_vfmadd_s(_r12, _k12, _sum0);
                _sum0 = __lsx_vfmadd_s(_r13, _k13, _sum0);
                _sum0 = __lsx_vfmadd_s(_r14, _k14, _sum0);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);
                __m128 _r23 = (__m128)__lsx_vld(r2 + 4 * 3, 0);
                __m128 _r24 = (__m128)__lsx_vld(r2 + 4 * 4, 0);

                _sum1 = __lsx_vfmadd_s(_r20, _k10, _sum1);
                _sum1 = __lsx_vfmadd_s(_r21, _k11, _sum1);
                _sum1 = __lsx_vfmadd_s(_r22, _k12, _sum1);
                _sum1 = __lsx_vfmadd_s(_r23, _k13, _sum1);
                _sum1 = __lsx_vfmadd_s(_r24, _k14, _sum1);

                __m128 _k20 = (__m128)__lsx_vld(k0, 0);
                __m128 _k21 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k22 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k23 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k24 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r20, _k20, _sum0);
                _sum0 = __lsx_vfmadd_s(_r21, _k21, _sum0);
                _sum0 = __lsx_vfmadd_s(_r22, _k22, _sum0);
                _sum0 = __lsx_vfmadd_s(_r23, _k23, _sum0);
                _sum0 = __lsx_vfmadd_s(_r24, _k24, _sum0);

                __m128 _r30 = (__m128)__lsx_vld(r3, 0);
                __m128 _r31 = (__m128)__lsx_vld(r3 + 4, 0);
                __m128 _r32 = (__m128)__lsx_vld(r3 + 4 * 2, 0);
                __m128 _r33 = (__m128)__lsx_vld(r3 + 4 * 3, 0);
                __m128 _r34 = (__m128)__lsx_vld(r3 + 4 * 4, 0);

                _sum1 = __lsx_vfmadd_s(_r30, _k20, _sum1);
                _sum1 = __lsx_vfmadd_s(_r31, _k21, _sum1);
                _sum1 = __lsx_vfmadd_s(_r32, _k22, _sum1);
                _sum1 = __lsx_vfmadd_s(_r33, _k23, _sum1);
                _sum1 = __lsx_vfmadd_s(_r34, _k24, _sum1);

                __m128 _k30 = (__m128)__lsx_vld(k0, 0);
                __m128 _k31 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k32 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k33 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k34 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r30, _k30, _sum0);
                _sum0 = __lsx_vfmadd_s(_r31, _k31, _sum0);
                _sum0 = __lsx_vfmadd_s(_r32, _k32, _sum0);
                _sum0 = __lsx_vfmadd_s(_r33, _k33, _sum0);
                _sum0 = __lsx_vfmadd_s(_r34, _k34, _sum0);

                __m128 _r40 = (__m128)__lsx_vld(r4, 0);
                __m128 _r41 = (__m128)__lsx_vld(r4 + 4, 0);
                __m128 _r42 = (__m128)__lsx_vld(r4 + 4 * 2, 0);
                __m128 _r43 = (__m128)__lsx_vld(r4 + 4 * 3, 0);
                __m128 _r44 = (__m128)__lsx_vld(r4 + 4 * 4, 0);

                _sum1 = __lsx_vfmadd_s(_r40, _k30, _sum1);
                _sum1 = __lsx_vfmadd_s(_r41, _k31, _sum1);
                _sum1 = __lsx_vfmadd_s(_r42, _k32, _sum1);
                _sum1 = __lsx_vfmadd_s(_r43, _k33, _sum1);
                _sum1 = __lsx_vfmadd_s(_r44, _k34, _sum1);

                __m128 _k40 = (__m128)__lsx_vld(k0, 0);
                __m128 _k41 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k42 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k43 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k44 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 -= 4 * 20;

                _sum0 = __lsx_vfmadd_s(_r40, _k40, _sum0);
                _sum0 = __lsx_vfmadd_s(_r41, _k41, _sum0);
                _sum0 = __lsx_vfmadd_s(_r42, _k42, _sum0);
                _sum0 = __lsx_vfmadd_s(_r43, _k43, _sum0);
                _sum0 = __lsx_vfmadd_s(_r44, _k44, _sum0);

                __m128 _r50 = (__m128)__lsx_vld(r5, 0);
                __m128 _r51 = (__m128)__lsx_vld(r5 + 4, 0);
                __m128 _r52 = (__m128)__lsx_vld(r5 + 4 * 2, 0);
                __m128 _r53 = (__m128)__lsx_vld(r5 + 4 * 3, 0);
                __m128 _r54 = (__m128)__lsx_vld(r5 + 4 * 4, 0);

                _sum1 = __lsx_vfmadd_s(_r50, _k40, _sum1);
                _sum1 = __lsx_vfmadd_s(_r51, _k41, _sum1);
                _sum1 = __lsx_vfmadd_s(_r52, _k42, _sum1);
                _sum1 = __lsx_vfmadd_s(_r53, _k43, _sum1);
                _sum1 = __lsx_vfmadd_s(_r54, _k44, _sum1);

                __lsx_vst(_sum0, outptr0, 0);
                __lsx_vst(_sum1, outptr1, 0);

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

                __m128 _sum0 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);
                __m128 _r04 = (__m128)__lsx_vld(r0 + 4 * 4, 0);

                __m128 _k00 = (__m128)__lsx_vld(k0, 0);
                __m128 _k01 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k02 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k03 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k04 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r00, _k00, _sum0);
                _sum0 = __lsx_vfmadd_s(_r01, _k01, _sum0);
                _sum0 = __lsx_vfmadd_s(_r02, _k02, _sum0);
                _sum0 = __lsx_vfmadd_s(_r03, _k03, _sum0);
                _sum0 = __lsx_vfmadd_s(_r04, _k04, _sum0);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);
                __m128 _r13 = (__m128)__lsx_vld(r1 + 4 * 3, 0);
                __m128 _r14 = (__m128)__lsx_vld(r1 + 4 * 4, 0);

                __m128 _k10 = (__m128)__lsx_vld(k0, 0);
                __m128 _k11 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k12 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k13 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k14 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r10, _k10, _sum0);
                _sum0 = __lsx_vfmadd_s(_r11, _k11, _sum0);
                _sum0 = __lsx_vfmadd_s(_r12, _k12, _sum0);
                _sum0 = __lsx_vfmadd_s(_r13, _k13, _sum0);
                _sum0 = __lsx_vfmadd_s(_r14, _k14, _sum0);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);
                __m128 _r23 = (__m128)__lsx_vld(r2 + 4 * 3, 0);
                __m128 _r24 = (__m128)__lsx_vld(r2 + 4 * 4, 0);

                __m128 _k20 = (__m128)__lsx_vld(k0, 0);
                __m128 _k21 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k22 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k23 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k24 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r20, _k20, _sum0);
                _sum0 = __lsx_vfmadd_s(_r21, _k21, _sum0);
                _sum0 = __lsx_vfmadd_s(_r22, _k22, _sum0);
                _sum0 = __lsx_vfmadd_s(_r23, _k23, _sum0);
                _sum0 = __lsx_vfmadd_s(_r24, _k24, _sum0);

                __m128 _r30 = (__m128)__lsx_vld(r3, 0);
                __m128 _r31 = (__m128)__lsx_vld(r3 + 4, 0);
                __m128 _r32 = (__m128)__lsx_vld(r3 + 4 * 2, 0);
                __m128 _r33 = (__m128)__lsx_vld(r3 + 4 * 3, 0);
                __m128 _r34 = (__m128)__lsx_vld(r3 + 4 * 4, 0);

                __m128 _k30 = (__m128)__lsx_vld(k0, 0);
                __m128 _k31 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k32 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k33 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k34 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r30, _k30, _sum0);
                _sum0 = __lsx_vfmadd_s(_r31, _k31, _sum0);
                _sum0 = __lsx_vfmadd_s(_r32, _k32, _sum0);
                _sum0 = __lsx_vfmadd_s(_r33, _k33, _sum0);
                _sum0 = __lsx_vfmadd_s(_r34, _k34, _sum0);

                __m128 _r40 = (__m128)__lsx_vld(r4, 0);
                __m128 _r41 = (__m128)__lsx_vld(r4 + 4, 0);
                __m128 _r42 = (__m128)__lsx_vld(r4 + 4 * 2, 0);
                __m128 _r43 = (__m128)__lsx_vld(r4 + 4 * 3, 0);
                __m128 _r44 = (__m128)__lsx_vld(r4 + 4 * 4, 0);

                __m128 _k40 = (__m128)__lsx_vld(k0, 0);
                __m128 _k41 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k42 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k43 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k44 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 -= 4 * 20;

                _sum0 = __lsx_vfmadd_s(_r40, _k40, _sum0);
                _sum0 = __lsx_vfmadd_s(_r41, _k41, _sum0);
                _sum0 = __lsx_vfmadd_s(_r42, _k42, _sum0);
                _sum0 = __lsx_vfmadd_s(_r43, _k43, _sum0);
                _sum0 = __lsx_vfmadd_s(_r44, _k44, _sum0);

                __lsx_vst(_sum0, outptr0, 0);

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

static void convdw5x5s2_pack4_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

                __m128 _sum0 = _bias0;

                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);
                __m128 _r04 = (__m128)__lsx_vld(r0 + 4 * 4, 0);

                __m128 _k00 = (__m128)__lsx_vld(k0, 0);
                __m128 _k01 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k02 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k03 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k04 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r00, _k00, _sum0);
                _sum0 = __lsx_vfmadd_s(_r01, _k01, _sum0);
                _sum0 = __lsx_vfmadd_s(_r02, _k02, _sum0);
                _sum0 = __lsx_vfmadd_s(_r03, _k03, _sum0);
                _sum0 = __lsx_vfmadd_s(_r04, _k04, _sum0);

                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 4 * 2, 0);
                __m128 _r13 = (__m128)__lsx_vld(r1 + 4 * 3, 0);
                __m128 _r14 = (__m128)__lsx_vld(r1 + 4 * 4, 0);

                __m128 _k10 = (__m128)__lsx_vld(k0, 0);
                __m128 _k11 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k12 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k13 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k14 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r10, _k10, _sum0);
                _sum0 = __lsx_vfmadd_s(_r11, _k11, _sum0);
                _sum0 = __lsx_vfmadd_s(_r12, _k12, _sum0);
                _sum0 = __lsx_vfmadd_s(_r13, _k13, _sum0);
                _sum0 = __lsx_vfmadd_s(_r14, _k14, _sum0);

                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 4 * 2, 0);
                __m128 _r23 = (__m128)__lsx_vld(r2 + 4 * 3, 0);
                __m128 _r24 = (__m128)__lsx_vld(r2 + 4 * 4, 0);

                __m128 _k20 = (__m128)__lsx_vld(k0, 0);
                __m128 _k21 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k22 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k23 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k24 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r20, _k20, _sum0);
                _sum0 = __lsx_vfmadd_s(_r21, _k21, _sum0);
                _sum0 = __lsx_vfmadd_s(_r22, _k22, _sum0);
                _sum0 = __lsx_vfmadd_s(_r23, _k23, _sum0);
                _sum0 = __lsx_vfmadd_s(_r24, _k24, _sum0);

                __m128 _r30 = (__m128)__lsx_vld(r3, 0);
                __m128 _r31 = (__m128)__lsx_vld(r3 + 4, 0);
                __m128 _r32 = (__m128)__lsx_vld(r3 + 4 * 2, 0);
                __m128 _r33 = (__m128)__lsx_vld(r3 + 4 * 3, 0);
                __m128 _r34 = (__m128)__lsx_vld(r3 + 4 * 4, 0);

                __m128 _k30 = (__m128)__lsx_vld(k0, 0);
                __m128 _k31 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k32 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k33 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k34 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 += 4 * 5;

                _sum0 = __lsx_vfmadd_s(_r30, _k30, _sum0);
                _sum0 = __lsx_vfmadd_s(_r31, _k31, _sum0);
                _sum0 = __lsx_vfmadd_s(_r32, _k32, _sum0);
                _sum0 = __lsx_vfmadd_s(_r33, _k33, _sum0);
                _sum0 = __lsx_vfmadd_s(_r34, _k34, _sum0);

                __m128 _r40 = (__m128)__lsx_vld(r4, 0);
                __m128 _r41 = (__m128)__lsx_vld(r4 + 4, 0);
                __m128 _r42 = (__m128)__lsx_vld(r4 + 4 * 2, 0);
                __m128 _r43 = (__m128)__lsx_vld(r4 + 4 * 3, 0);
                __m128 _r44 = (__m128)__lsx_vld(r4 + 4 * 4, 0);

                __m128 _k40 = (__m128)__lsx_vld(k0, 0);
                __m128 _k41 = (__m128)__lsx_vld(k0 + 4, 0);
                __m128 _k42 = (__m128)__lsx_vld(k0 + 4 * 2, 0);
                __m128 _k43 = (__m128)__lsx_vld(k0 + 4 * 3, 0);
                __m128 _k44 = (__m128)__lsx_vld(k0 + 4 * 4, 0);
                k0 -= 4 * 20;

                _sum0 = __lsx_vfmadd_s(_r40, _k40, _sum0);
                _sum0 = __lsx_vfmadd_s(_r41, _k41, _sum0);
                _sum0 = __lsx_vfmadd_s(_r42, _k42, _sum0);
                _sum0 = __lsx_vfmadd_s(_r43, _k43, _sum0);
                _sum0 = __lsx_vfmadd_s(_r44, _k44, _sum0);

                __lsx_vst(_sum0, outptr0, 0);

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

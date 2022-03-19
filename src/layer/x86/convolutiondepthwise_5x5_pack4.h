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

static void convdw5x5s1_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        __m128 _bias0 = bias ? _mm_loadu_ps(bias + g * 4) : _mm_setzero_ps();

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
                __m128 _sum0 = _bias0;
                __m128 _sum1 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                __m128 _r04 = _mm_load_ps(r0 + 4 * 4);

                __m128 _k00 = _mm_load_ps(k0);
                __m128 _k01 = _mm_load_ps(k0 + 4);
                __m128 _k02 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k03 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k04 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k03, _r03, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k04, _r04, _sum0);

                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 4 * 2);
                __m128 _r13 = _mm_load_ps(r1 + 4 * 3);
                __m128 _r14 = _mm_load_ps(r1 + 4 * 4);

                _sum1 = _mm_comp_fmadd_ps(_k00, _r10, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k01, _r11, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k02, _r12, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k03, _r13, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k04, _r14, _sum1);

                __m128 _k10 = _mm_load_ps(k0);
                __m128 _k11 = _mm_load_ps(k0 + 4);
                __m128 _k12 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k13 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k14 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k13, _r13, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k14, _r14, _sum0);

                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 4 * 2);
                __m128 _r23 = _mm_load_ps(r2 + 4 * 3);
                __m128 _r24 = _mm_load_ps(r2 + 4 * 4);

                _sum1 = _mm_comp_fmadd_ps(_k10, _r20, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k11, _r21, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k12, _r22, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k13, _r23, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k14, _r24, _sum1);

                __m128 _k20 = _mm_load_ps(k0);
                __m128 _k21 = _mm_load_ps(k0 + 4);
                __m128 _k22 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k23 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k24 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k23, _r23, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k24, _r24, _sum0);

                __m128 _r30 = _mm_load_ps(r3);
                __m128 _r31 = _mm_load_ps(r3 + 4);
                __m128 _r32 = _mm_load_ps(r3 + 4 * 2);
                __m128 _r33 = _mm_load_ps(r3 + 4 * 3);
                __m128 _r34 = _mm_load_ps(r3 + 4 * 4);

                _sum1 = _mm_comp_fmadd_ps(_k20, _r30, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k21, _r31, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k22, _r32, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k23, _r33, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k24, _r34, _sum1);

                __m128 _k30 = _mm_load_ps(k0);
                __m128 _k31 = _mm_load_ps(k0 + 4);
                __m128 _k32 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k33 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k34 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k30, _r30, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k31, _r31, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k32, _r32, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k33, _r33, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k34, _r34, _sum0);

                __m128 _r40 = _mm_load_ps(r4);
                __m128 _r41 = _mm_load_ps(r4 + 4);
                __m128 _r42 = _mm_load_ps(r4 + 4 * 2);
                __m128 _r43 = _mm_load_ps(r4 + 4 * 3);
                __m128 _r44 = _mm_load_ps(r4 + 4 * 4);

                _sum1 = _mm_comp_fmadd_ps(_k30, _r40, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k31, _r41, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k32, _r42, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k33, _r43, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k34, _r44, _sum1);

                __m128 _k40 = _mm_load_ps(k0);
                __m128 _k41 = _mm_load_ps(k0 + 4);
                __m128 _k42 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k43 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k44 = _mm_load_ps(k0 + 4 * 4);
                k0 -= 4 * 20;

                _sum0 = _mm_comp_fmadd_ps(_k40, _r40, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k41, _r41, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k42, _r42, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k43, _r43, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k44, _r44, _sum0);

                __m128 _r50 = _mm_load_ps(r5);
                __m128 _r51 = _mm_load_ps(r5 + 4);
                __m128 _r52 = _mm_load_ps(r5 + 4 * 2);
                __m128 _r53 = _mm_load_ps(r5 + 4 * 3);
                __m128 _r54 = _mm_load_ps(r5 + 4 * 4);

                _sum1 = _mm_comp_fmadd_ps(_k40, _r50, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k41, _r51, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k42, _r52, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k43, _r53, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_k44, _r54, _sum1);

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr1, _sum1);

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
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                __m128 _r04 = _mm_load_ps(r0 + 4 * 4);

                __m128 _k00 = _mm_load_ps(k0);
                __m128 _k01 = _mm_load_ps(k0 + 4);
                __m128 _k02 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k03 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k04 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k03, _r03, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k04, _r04, _sum0);

                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 4 * 2);
                __m128 _r13 = _mm_load_ps(r1 + 4 * 3);
                __m128 _r14 = _mm_load_ps(r1 + 4 * 4);

                __m128 _k10 = _mm_load_ps(k0);
                __m128 _k11 = _mm_load_ps(k0 + 4);
                __m128 _k12 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k13 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k14 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k13, _r13, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k14, _r14, _sum0);

                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 4 * 2);
                __m128 _r23 = _mm_load_ps(r2 + 4 * 3);
                __m128 _r24 = _mm_load_ps(r2 + 4 * 4);

                __m128 _k20 = _mm_load_ps(k0);
                __m128 _k21 = _mm_load_ps(k0 + 4);
                __m128 _k22 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k23 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k24 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k23, _r23, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k24, _r24, _sum0);

                __m128 _r30 = _mm_load_ps(r3);
                __m128 _r31 = _mm_load_ps(r3 + 4);
                __m128 _r32 = _mm_load_ps(r3 + 4 * 2);
                __m128 _r33 = _mm_load_ps(r3 + 4 * 3);
                __m128 _r34 = _mm_load_ps(r3 + 4 * 4);

                __m128 _k30 = _mm_load_ps(k0);
                __m128 _k31 = _mm_load_ps(k0 + 4);
                __m128 _k32 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k33 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k34 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k30, _r30, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k31, _r31, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k32, _r32, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k33, _r33, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k34, _r34, _sum0);

                __m128 _r40 = _mm_load_ps(r4);
                __m128 _r41 = _mm_load_ps(r4 + 4);
                __m128 _r42 = _mm_load_ps(r4 + 4 * 2);
                __m128 _r43 = _mm_load_ps(r4 + 4 * 3);
                __m128 _r44 = _mm_load_ps(r4 + 4 * 4);

                __m128 _k40 = _mm_load_ps(k0);
                __m128 _k41 = _mm_load_ps(k0 + 4);
                __m128 _k42 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k43 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k44 = _mm_load_ps(k0 + 4 * 4);
                k0 -= 4 * 20;

                _sum0 = _mm_comp_fmadd_ps(_k40, _r40, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k41, _r41, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k42, _r42, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k43, _r43, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k44, _r44, _sum0);

                _mm_store_ps(outptr0, _sum0);

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

static void convdw5x5s2_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        __m128 _bias0 = bias ? _mm_loadu_ps(bias + g * 4) : _mm_setzero_ps();

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
                __m128 _sum0 = _bias0;

                __m128 _r00 = _mm_load_ps(r0);
                __m128 _r01 = _mm_load_ps(r0 + 4);
                __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                __m128 _r04 = _mm_load_ps(r0 + 4 * 4);

                __m128 _k00 = _mm_load_ps(k0);
                __m128 _k01 = _mm_load_ps(k0 + 4);
                __m128 _k02 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k03 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k04 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k03, _r03, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k04, _r04, _sum0);

                __m128 _r10 = _mm_load_ps(r1);
                __m128 _r11 = _mm_load_ps(r1 + 4);
                __m128 _r12 = _mm_load_ps(r1 + 4 * 2);
                __m128 _r13 = _mm_load_ps(r1 + 4 * 3);
                __m128 _r14 = _mm_load_ps(r1 + 4 * 4);

                __m128 _k10 = _mm_load_ps(k0);
                __m128 _k11 = _mm_load_ps(k0 + 4);
                __m128 _k12 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k13 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k14 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k13, _r13, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k14, _r14, _sum0);

                __m128 _r20 = _mm_load_ps(r2);
                __m128 _r21 = _mm_load_ps(r2 + 4);
                __m128 _r22 = _mm_load_ps(r2 + 4 * 2);
                __m128 _r23 = _mm_load_ps(r2 + 4 * 3);
                __m128 _r24 = _mm_load_ps(r2 + 4 * 4);

                __m128 _k20 = _mm_load_ps(k0);
                __m128 _k21 = _mm_load_ps(k0 + 4);
                __m128 _k22 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k23 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k24 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k23, _r23, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k24, _r24, _sum0);

                __m128 _r30 = _mm_load_ps(r3);
                __m128 _r31 = _mm_load_ps(r3 + 4);
                __m128 _r32 = _mm_load_ps(r3 + 4 * 2);
                __m128 _r33 = _mm_load_ps(r3 + 4 * 3);
                __m128 _r34 = _mm_load_ps(r3 + 4 * 4);

                __m128 _k30 = _mm_load_ps(k0);
                __m128 _k31 = _mm_load_ps(k0 + 4);
                __m128 _k32 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k33 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k34 = _mm_load_ps(k0 + 4 * 4);
                k0 += 4 * 5;

                _sum0 = _mm_comp_fmadd_ps(_k30, _r30, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k31, _r31, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k32, _r32, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k33, _r33, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k34, _r34, _sum0);

                __m128 _r40 = _mm_load_ps(r4);
                __m128 _r41 = _mm_load_ps(r4 + 4);
                __m128 _r42 = _mm_load_ps(r4 + 4 * 2);
                __m128 _r43 = _mm_load_ps(r4 + 4 * 3);
                __m128 _r44 = _mm_load_ps(r4 + 4 * 4);

                __m128 _k40 = _mm_load_ps(k0);
                __m128 _k41 = _mm_load_ps(k0 + 4);
                __m128 _k42 = _mm_load_ps(k0 + 4 * 2);
                __m128 _k43 = _mm_load_ps(k0 + 4 * 3);
                __m128 _k44 = _mm_load_ps(k0 + 4 * 4);
                k0 -= 4 * 20;

                _sum0 = _mm_comp_fmadd_ps(_k40, _r40, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k41, _r41, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k42, _r42, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k43, _r43, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_k44, _r44, _sum0);

                _mm_store_ps(outptr0, _sum0);

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

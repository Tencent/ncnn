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

static void convdw5x5s1_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
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
                __m512 _sum0 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);
                __m512 _r03 = _mm512_load_ps(r0 + 48);
                __m512 _r04 = _mm512_load_ps(r0 + 64);

                __m512 _k00 = _mm512_load_ps(k0);
                __m512 _k01 = _mm512_load_ps(k0 + 16);
                __m512 _k02 = _mm512_load_ps(k0 + 32);
                __m512 _k03 = _mm512_load_ps(k0 + 48);
                __m512 _k04 = _mm512_load_ps(k0 + 64);
                k0 += 80;

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm512_fmadd_ps(_k03, _r03, _sum0);
                _sum0 = _mm512_fmadd_ps(_k04, _r04, _sum0);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);
                __m512 _r13 = _mm512_load_ps(r1 + 48);
                __m512 _r14 = _mm512_load_ps(r1 + 64);

                __m512 _k10 = _mm512_load_ps(k0);
                __m512 _k11 = _mm512_load_ps(k0 + 16);
                __m512 _k12 = _mm512_load_ps(k0 + 32);
                __m512 _k13 = _mm512_load_ps(k0 + 48);
                __m512 _k14 = _mm512_load_ps(k0 + 64);
                k0 += 80;

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm512_fmadd_ps(_k13, _r13, _sum0);
                _sum0 = _mm512_fmadd_ps(_k14, _r14, _sum0);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);
                __m512 _r23 = _mm512_load_ps(r2 + 48);
                __m512 _r24 = _mm512_load_ps(r2 + 64);

                __m512 _k20 = _mm512_load_ps(k0);
                __m512 _k21 = _mm512_load_ps(k0 + 16);
                __m512 _k22 = _mm512_load_ps(k0 + 32);
                __m512 _k23 = _mm512_load_ps(k0 + 48);
                __m512 _k24 = _mm512_load_ps(k0 + 64);
                k0 += 80;

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);
                _sum0 = _mm512_fmadd_ps(_k23, _r23, _sum0);
                _sum0 = _mm512_fmadd_ps(_k24, _r24, _sum0);

                __m512 _r30 = _mm512_load_ps(r3);
                __m512 _r31 = _mm512_load_ps(r3 + 16);
                __m512 _r32 = _mm512_load_ps(r3 + 32);
                __m512 _r33 = _mm512_load_ps(r3 + 48);
                __m512 _r34 = _mm512_load_ps(r3 + 64);

                __m512 _k30 = _mm512_load_ps(k0);
                __m512 _k31 = _mm512_load_ps(k0 + 16);
                __m512 _k32 = _mm512_load_ps(k0 + 32);
                __m512 _k33 = _mm512_load_ps(k0 + 48);
                __m512 _k34 = _mm512_load_ps(k0 + 64);
                k0 += 80;

                _sum0 = _mm512_fmadd_ps(_k30, _r30, _sum0);
                _sum0 = _mm512_fmadd_ps(_k31, _r31, _sum0);
                _sum0 = _mm512_fmadd_ps(_k32, _r32, _sum0);
                _sum0 = _mm512_fmadd_ps(_k33, _r33, _sum0);
                _sum0 = _mm512_fmadd_ps(_k34, _r34, _sum0);

                __m512 _r40 = _mm512_load_ps(r4);
                __m512 _r41 = _mm512_load_ps(r4 + 16);
                __m512 _r42 = _mm512_load_ps(r4 + 32);
                __m512 _r43 = _mm512_load_ps(r4 + 48);
                __m512 _r44 = _mm512_load_ps(r4 + 64);

                __m512 _k40 = _mm512_load_ps(k0);
                __m512 _k41 = _mm512_load_ps(k0 + 16);
                __m512 _k42 = _mm512_load_ps(k0 + 32);
                __m512 _k43 = _mm512_load_ps(k0 + 48);
                __m512 _k44 = _mm512_load_ps(k0 + 64);
                k0 -= 320;

                _sum0 = _mm512_fmadd_ps(_k40, _r40, _sum0);
                _sum0 = _mm512_fmadd_ps(_k41, _r41, _sum0);
                _sum0 = _mm512_fmadd_ps(_k42, _r42, _sum0);
                _sum0 = _mm512_fmadd_ps(_k43, _r43, _sum0);
                _sum0 = _mm512_fmadd_ps(_k44, _r44, _sum0);

                _mm512_store_ps(outptr0, _sum0);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                r4 += 16;
                outptr0 += 16;
            }

            r0 += 4 * 16;
            r1 += 4 * 16;
            r2 += 4 * 16;
            r3 += 4 * 16;
            r4 += 4 * 16;
        }
    }
}

static void convdw5x5s2_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
        const float* r3 = img0.row(3);
        const float* r4 = img0.row(4);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;

            for (; j < outw; j++)
            {
                __m512 _sum0 = _bias0;

                __m512 _r00 = _mm512_load_ps(r0);
                __m512 _r01 = _mm512_load_ps(r0 + 16);
                __m512 _r02 = _mm512_load_ps(r0 + 32);
                __m512 _r03 = _mm512_load_ps(r0 + 48);
                __m512 _r04 = _mm512_load_ps(r0 + 64);

                __m512 _k00 = _mm512_load_ps(k0);
                __m512 _k01 = _mm512_load_ps(k0 + 16);
                __m512 _k02 = _mm512_load_ps(k0 + 32);
                __m512 _k03 = _mm512_load_ps(k0 + 48);
                __m512 _k04 = _mm512_load_ps(k0 + 64);
                k0 += 80;

                _sum0 = _mm512_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm512_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm512_fmadd_ps(_k02, _r02, _sum0);
                _sum0 = _mm512_fmadd_ps(_k03, _r03, _sum0);
                _sum0 = _mm512_fmadd_ps(_k04, _r04, _sum0);

                __m512 _r10 = _mm512_load_ps(r1);
                __m512 _r11 = _mm512_load_ps(r1 + 16);
                __m512 _r12 = _mm512_load_ps(r1 + 32);
                __m512 _r13 = _mm512_load_ps(r1 + 48);
                __m512 _r14 = _mm512_load_ps(r1 + 64);

                __m512 _k10 = _mm512_load_ps(k0);
                __m512 _k11 = _mm512_load_ps(k0 + 16);
                __m512 _k12 = _mm512_load_ps(k0 + 32);
                __m512 _k13 = _mm512_load_ps(k0 + 48);
                __m512 _k14 = _mm512_load_ps(k0 + 64);
                k0 += 80;

                _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm512_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm512_fmadd_ps(_k12, _r12, _sum0);
                _sum0 = _mm512_fmadd_ps(_k13, _r13, _sum0);
                _sum0 = _mm512_fmadd_ps(_k14, _r14, _sum0);

                __m512 _r20 = _mm512_load_ps(r2);
                __m512 _r21 = _mm512_load_ps(r2 + 16);
                __m512 _r22 = _mm512_load_ps(r2 + 32);
                __m512 _r23 = _mm512_load_ps(r2 + 48);
                __m512 _r24 = _mm512_load_ps(r2 + 64);

                __m512 _k20 = _mm512_load_ps(k0);
                __m512 _k21 = _mm512_load_ps(k0 + 16);
                __m512 _k22 = _mm512_load_ps(k0 + 32);
                __m512 _k23 = _mm512_load_ps(k0 + 48);
                __m512 _k24 = _mm512_load_ps(k0 + 64);
                k0 += 80;

                _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm512_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm512_fmadd_ps(_k22, _r22, _sum0);
                _sum0 = _mm512_fmadd_ps(_k23, _r23, _sum0);
                _sum0 = _mm512_fmadd_ps(_k24, _r24, _sum0);

                __m512 _r30 = _mm512_load_ps(r3);
                __m512 _r31 = _mm512_load_ps(r3 + 16);
                __m512 _r32 = _mm512_load_ps(r3 + 32);
                __m512 _r33 = _mm512_load_ps(r3 + 48);
                __m512 _r34 = _mm512_load_ps(r3 + 64);

                __m512 _k30 = _mm512_load_ps(k0);
                __m512 _k31 = _mm512_load_ps(k0 + 16);
                __m512 _k32 = _mm512_load_ps(k0 + 32);
                __m512 _k33 = _mm512_load_ps(k0 + 48);
                __m512 _k34 = _mm512_load_ps(k0 + 64);
                k0 += 80;

                _sum0 = _mm512_fmadd_ps(_k30, _r30, _sum0);
                _sum0 = _mm512_fmadd_ps(_k31, _r31, _sum0);
                _sum0 = _mm512_fmadd_ps(_k32, _r32, _sum0);
                _sum0 = _mm512_fmadd_ps(_k33, _r33, _sum0);
                _sum0 = _mm512_fmadd_ps(_k34, _r34, _sum0);

                __m512 _r40 = _mm512_load_ps(r4);
                __m512 _r41 = _mm512_load_ps(r4 + 16);
                __m512 _r42 = _mm512_load_ps(r4 + 32);
                __m512 _r43 = _mm512_load_ps(r4 + 48);
                __m512 _r44 = _mm512_load_ps(r4 + 64);

                __m512 _k40 = _mm512_load_ps(k0);
                __m512 _k41 = _mm512_load_ps(k0 + 16);
                __m512 _k42 = _mm512_load_ps(k0 + 32);
                __m512 _k43 = _mm512_load_ps(k0 + 48);
                __m512 _k44 = _mm512_load_ps(k0 + 64);
                k0 -= 320;

                _sum0 = _mm512_fmadd_ps(_k40, _r40, _sum0);
                _sum0 = _mm512_fmadd_ps(_k41, _r41, _sum0);
                _sum0 = _mm512_fmadd_ps(_k42, _r42, _sum0);
                _sum0 = _mm512_fmadd_ps(_k43, _r43, _sum0);
                _sum0 = _mm512_fmadd_ps(_k44, _r44, _sum0);

                _mm512_store_ps(outptr0, _sum0);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                r3 += 32;
                r4 += 32;
                outptr0 += 16;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }
}

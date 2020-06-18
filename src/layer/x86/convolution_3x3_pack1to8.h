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

static void conv3x3s1_pack1to8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;
    const float* bias = _bias;

    int nn_outch = 0;
    int remain_outch_start = 0;

    
    nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_set1_ps(0.f);
        __m256 _bias1 = bias ? _mm256_loadu_ps((const float*)bias + (p + 1) * 8) : _mm256_set1_ps(0.f);
        out0.fill(_bias0);
        out1.fill(_bias1);

        const float* k0 = kernel.channel(p);
        const float* k1 = kernel.channel(p + 1);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m256 _k00_0 = _mm256_loadu_ps(k0);
            __m256 _k01_0 = _mm256_loadu_ps(k0 + 8);
            __m256 _k02_0 = _mm256_loadu_ps(k0 + 16);
            __m256 _k10_0 = _mm256_loadu_ps(k0 + 24);
            __m256 _k11_0 = _mm256_loadu_ps(k0 + 32);
            __m256 _k12_0 = _mm256_loadu_ps(k0 + 40);
            __m256 _k20_0 = _mm256_loadu_ps(k0 + 48);
            __m256 _k21_0 = _mm256_loadu_ps(k0 + 56);
            __m256 _k22_0 = _mm256_loadu_ps(k0 + 64);

            __m256 _k00_1 = _mm256_loadu_ps(k1);
            __m256 _k01_1 = _mm256_loadu_ps(k1 + 8);
            __m256 _k02_1 = _mm256_loadu_ps(k1 + 16);
            __m256 _k10_1 = _mm256_loadu_ps(k1 + 24);
            __m256 _k11_1 = _mm256_loadu_ps(k1 + 32);
            __m256 _k12_1 = _mm256_loadu_ps(k1 + 40);
            __m256 _k20_1 = _mm256_loadu_ps(k1 + 48);
            __m256 _k21_1 = _mm256_loadu_ps(k1 + 56);
            __m256 _k22_1 = _mm256_loadu_ps(k1 + 64);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

                for (; j < outw; j++)
                {
                    __m256 _sum00 = _mm256_loadu_ps(outptr0);
                    __m256 _sum10 = _mm256_loadu_ps(outptr1);

                    __m256 _r01 = _mm256_broadcast_ss(r0);
                    __m256 _r02 = _mm256_broadcast_ss(r0+1);
                    __m256 _r03 = _mm256_broadcast_ss(r0+2);
                    __m256 _r11 = _mm256_broadcast_ss(r1);
                    __m256 _r12 = _mm256_broadcast_ss(r1+1);
                    __m256 _r13 = _mm256_broadcast_ss(r1+2);
                    __m256 _r21 = _mm256_broadcast_ss(r2);
                    __m256 _r22 = _mm256_broadcast_ss(r2+1);
                    __m256 _r23 = _mm256_broadcast_ss(r2+2);

                    _sum00 = _mm256_fmadd_ps(_r01,_k00_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r02,_k01_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r03,_k02_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r11,_k10_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r12,_k11_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r13,_k12_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r21,_k20_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r22,_k21_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r23,_k22_0,_sum00);


                    _sum10 = _mm256_fmadd_ps(_r01,_k00_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r02,_k01_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r03,_k02_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r11,_k10_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r12,_k11_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r13,_k12_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r21,_k20_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r22,_k21_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r23,_k22_1,_sum10);

                    _mm256_storeu_ps(outptr0, _sum00);
                    _mm256_storeu_ps(outptr1, _sum10);

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                    outptr0 += 8;
                    outptr1 += 8;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * 8;
            k1 += 9 * 8;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_set1_ps(0.f);
        out0.fill(_bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m256 _k00 = _mm256_loadu_ps(k0);
            __m256 _k01 = _mm256_loadu_ps(k0 + 8);
            __m256 _k02 = _mm256_loadu_ps(k0 + 16);
            __m256 _k10 = _mm256_loadu_ps(k0 + 24);
            __m256 _k11 = _mm256_loadu_ps(k0 + 32);
            __m256 _k12 = _mm256_loadu_ps(k0 + 40);
            __m256 _k20 = _mm256_loadu_ps(k0 + 48);
            __m256 _k21 = _mm256_loadu_ps(k0 + 56);
            __m256 _k22 = _mm256_loadu_ps(k0 + 64);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j < outw; j++)
                {
                    __m256 _sum0 = _mm256_loadu_ps(outptr0);

                    __m256 _r01 = _mm256_broadcast_ss(r0);
                    __m256 _r02 = _mm256_broadcast_ss(r0 + 1);
                    __m256 _r03 = _mm256_broadcast_ss(r0 + 2);
                    __m256 _r11 = _mm256_broadcast_ss(r1);
                    __m256 _r12 = _mm256_broadcast_ss(r1 + 1);
                    __m256 _r13 = _mm256_broadcast_ss(r1 + 2);
                    __m256 _r21 = _mm256_broadcast_ss(r2);
                    __m256 _r22 = _mm256_broadcast_ss(r2 + 1);
                    __m256 _r23 = _mm256_broadcast_ss(r2 + 2);

                    _sum0 = _mm256_fmadd_ps(_r01,_k00,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r02,_k01,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r03,_k02,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r11,_k10,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r12,_k11,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r13,_k12,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r21,_k20,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r22,_k21,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r23,_k22,_sum0);


                    _mm256_storeu_ps(outptr0, _sum0);

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                    outptr0 += 8;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * 8;
        }
    }
}

static void conv3x3s2_pack1to8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_set1_ps(0.f);
        __m256 _bias1 = bias ? _mm256_loadu_ps((const float*)bias + (p + 1) * 8) : _mm256_set1_ps(0.f);
        out0.fill(_bias0);
        out1.fill(_bias1);

        const float* k0 = kernel.channel(p);
        const float* k1 = kernel.channel(p + 1);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m256 _k00_0 = _mm256_loadu_ps(k0);
            __m256 _k01_0 = _mm256_loadu_ps(k0 + 8);
            __m256 _k02_0 = _mm256_loadu_ps(k0 + 16);
            __m256 _k10_0 = _mm256_loadu_ps(k0 + 24);
            __m256 _k11_0 = _mm256_loadu_ps(k0 + 32);
            __m256 _k12_0 = _mm256_loadu_ps(k0 + 40);
            __m256 _k20_0 = _mm256_loadu_ps(k0 + 48);
            __m256 _k21_0 = _mm256_loadu_ps(k0 + 56);
            __m256 _k22_0 = _mm256_loadu_ps(k0 + 64);

            __m256 _k00_1 = _mm256_loadu_ps(k1);
            __m256 _k01_1 = _mm256_loadu_ps(k1 + 8);
            __m256 _k02_1 = _mm256_loadu_ps(k1 + 16);
            __m256 _k10_1 = _mm256_loadu_ps(k1 + 24);
            __m256 _k11_1 = _mm256_loadu_ps(k1 + 32);
            __m256 _k12_1 = _mm256_loadu_ps(k1 + 40);
            __m256 _k20_1 = _mm256_loadu_ps(k1 + 48);
            __m256 _k21_1 = _mm256_loadu_ps(k1 + 56);
            __m256 _k22_1 = _mm256_loadu_ps(k1 + 64);

            int i = 0;

            for (; i < outh; i++)
            {
                int remain = outw;
                for (; remain > 0; remain--)
                {
                    __m256 _sum00 = _mm256_loadu_ps(outptr0);
                    __m256 _sum10 = _mm256_loadu_ps(outptr1);

                    __m256 _r01 = _mm256_broadcast_ss(r0);
                    __m256 _r02 = _mm256_broadcast_ss(r0+1);
                    __m256 _r03 = _mm256_broadcast_ss(r0+2);
                    __m256 _r11 = _mm256_broadcast_ss(r1);
                    __m256 _r12 = _mm256_broadcast_ss(r1+1);
                    __m256 _r13 = _mm256_broadcast_ss(r1+2);
                    __m256 _r21 = _mm256_broadcast_ss(r2);
                    __m256 _r22 = _mm256_broadcast_ss(r2+1);
                    __m256 _r23 = _mm256_broadcast_ss(r2+2);

                    _sum00 = _mm256_fmadd_ps(_r01,_k00_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r02,_k01_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r03,_k02_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r11,_k10_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r12,_k11_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r13,_k12_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r21,_k20_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r22,_k21_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r23,_k22_0,_sum00);


                    _sum10 = _mm256_fmadd_ps(_r01,_k00_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r02,_k01_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r03,_k02_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r11,_k10_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r12,_k11_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r13,_k12_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r21,_k20_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r22,_k21_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r23,_k22_1,_sum10);

                    _mm256_storeu_ps(outptr0, _sum00);
                    _mm256_storeu_ps(outptr1, _sum10);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 8;
                    outptr1 += 8;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * 8;
            k1 += 9 * 8;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_set1_ps(0.f);
        out0.fill(_bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m256 _k00 = _mm256_loadu_ps(k0);
            __m256 _k01 = _mm256_loadu_ps(k0 + 8);
            __m256 _k02 = _mm256_loadu_ps(k0 + 16);
            __m256 _k10 = _mm256_loadu_ps(k0 + 24);
            __m256 _k11 = _mm256_loadu_ps(k0 + 32);
            __m256 _k12 = _mm256_loadu_ps(k0 + 40);
            __m256 _k20 = _mm256_loadu_ps(k0 + 48);
            __m256 _k21 = _mm256_loadu_ps(k0 + 56);
            __m256 _k22 = _mm256_loadu_ps(k0 + 64);

            int i = 0;

            for (; i < outh; i++)
            {
                int remain = outw;


                for (; remain > 0; remain--)
                {
                    __m256 _sum0 = _mm256_loadu_ps(outptr0);

                    __m256 _r01 = _mm256_broadcast_ss(r0);
                    __m256 _r02 = _mm256_broadcast_ss(r0+1);
                    __m256 _r03 = _mm256_broadcast_ss(r0+2);
                    __m256 _r11 = _mm256_broadcast_ss(r1);
                    __m256 _r12 = _mm256_broadcast_ss(r1+1);
                    __m256 _r13 = _mm256_broadcast_ss(r1+2);
                    __m256 _r21 = _mm256_broadcast_ss(r2);
                    __m256 _r22 = _mm256_broadcast_ss(r2+1);
                    __m256 _r23 = _mm256_broadcast_ss(r2+2);

                    _sum0 = _mm256_fmadd_ps(_r01,_k00,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r02,_k01,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r03,_k02,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r11,_k10,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r12,_k11,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r13,_k12,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r21,_k20,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r22,_k21,_sum0);
                    _sum0 = _mm256_fmadd_ps(_r23,_k22,_sum0);


                    _mm256_storeu_ps(outptr0, _sum0);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 8;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * 8;
        }
    }
}

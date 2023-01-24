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

static void conv3x3s1_pack16to1_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    int remain_outch_start = 0;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;
        out0.fill(bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            __m512 _k00 = _mm512_loadu_ps(k0);
            __m512 _k01 = _mm512_loadu_ps(k0 + 16);
            __m512 _k02 = _mm512_loadu_ps(k0 + 16 * 2);
            __m512 _k10 = _mm512_loadu_ps(k0 + 16 * 3);
            __m512 _k11 = _mm512_loadu_ps(k0 + 16 * 4);
            __m512 _k12 = _mm512_loadu_ps(k0 + 16 * 5);
            __m512 _k20 = _mm512_loadu_ps(k0 + 16 * 6);
            __m512 _k21 = _mm512_loadu_ps(k0 + 16 * 7);
            __m512 _k22 = _mm512_loadu_ps(k0 + 16 * 8);

            int i = 0;

            for (; i < outh; i++)
            {
                const float* r0 = img0.row(i);
                const float* r1 = img0.row(i + 1);
                const float* r2 = img0.row(i + 2);

                int j = 0;
                for (; j < outw; j++)
                {
                    __m512 _r00 = _mm512_loadu_ps(r0);
                    __m512 _r01 = _mm512_loadu_ps(r0 + 16);
                    __m512 _r02 = _mm512_loadu_ps(r0 + 32);

                    __m512 _sum0 = _mm512_mul_ps(_k00, _r00);
                    __m512 _sum1 = _mm512_mul_ps(_k01, _r01);
                    __m512 _sum2 = _mm512_mul_ps(_k02, _r02);

                    __m512 _r10 = _mm512_loadu_ps(r1);
                    __m512 _r11 = _mm512_loadu_ps(r1 + 16);
                    __m512 _r12 = _mm512_loadu_ps(r1 + 32);

                    _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                    _sum1 = _mm512_fmadd_ps(_k11, _r11, _sum1);
                    _sum2 = _mm512_fmadd_ps(_k12, _r12, _sum2);

                    __m512 _r20 = _mm512_loadu_ps(r2);
                    __m512 _r21 = _mm512_loadu_ps(r2 + 16);
                    __m512 _r22 = _mm512_loadu_ps(r2 + 32);

                    _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                    _sum1 = _mm512_fmadd_ps(_k21, _r21, _sum1);
                    _sum2 = _mm512_fmadd_ps(_k22, _r22, _sum2);

                    __m512 _sum = _mm512_add_ps(_sum0, _mm512_add_ps(_sum1, _sum2));

                    *outptr0 += _mm512_comp_reduce_add_ps(_sum);
                    outptr0++;
                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                }
            }

            k0 += 9 * 16;
        }
    }
}

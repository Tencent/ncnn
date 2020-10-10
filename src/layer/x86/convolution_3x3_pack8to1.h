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

static void conv3x3s1_pack8to1_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
                const float* r0 = img0.row(i);
                const float* r1 = img0.row(i + 1);
                const float* r2 = img0.row(i + 2);
                int j = 0;
                for (; j < outw; j++)
                {
                    __m256 _r00 = _mm256_loadu_ps(r0);
                    __m256 _r01 = _mm256_loadu_ps(r0 + 8);
                    __m256 _r02 = _mm256_loadu_ps(r0 + 16);

                    __m256 _sum0 = _mm256_mul_ps(_k00, _r00);
                    __m256 _sum1 = _mm256_mul_ps(_k01, _r01);
                    __m256 _sum2 = _mm256_mul_ps(_k02, _r02);

                    __m256 _r10 = _mm256_loadu_ps(r1);
                    __m256 _r11 = _mm256_loadu_ps(r1 + 8);
                    __m256 _r12 = _mm256_loadu_ps(r1 + 16);

                    _sum0 = _mm256_fmadd_ps(_k10, _r10, _sum0);
                    _sum1 = _mm256_fmadd_ps(_k11, _r11, _sum1);
                    _sum2 = _mm256_fmadd_ps(_k12, _r12, _sum2);

                    __m256 _r20 = _mm256_loadu_ps(r2);
                    __m256 _r21 = _mm256_loadu_ps(r2 + 8);
                    __m256 _r22 = _mm256_loadu_ps(r2 + 16);

                    _sum0 = _mm256_fmadd_ps(_k20, _r20, _sum0);
                    _sum1 = _mm256_fmadd_ps(_k21, _r21, _sum1);
                    _sum2 = _mm256_fmadd_ps(_k22, _r22, _sum2);
                    __m128 _sum = HorizontalSums(_sum0, _sum1, _sum2);

                    *outptr0 += _mm_reduce_add_ps(_sum); // dot
                    outptr0++;
                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
            }

            k0 += 9 * 8;
        }
    }
}

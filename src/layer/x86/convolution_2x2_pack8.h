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

static void conv2x2s1_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_set1_ps(0.f);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);

            const float* kptr = (const float*)kernel.channel(p).row(q);
            // const float* kptr = (const float*)kernel + 4 * inch * p * 64;

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 1 < outw; j += 2)
                {
                    __m256 _sum0 = _mm256_loadu_ps(outptr0);
                    __m256 _sum1 = _mm256_loadu_ps(outptr0 + 8);

                    __m256 _r00 = _mm256_broadcast_ss(r0);
                    __m256 _r01 = _mm256_broadcast_ss(r0 + 1);
                    __m256 _r02 = _mm256_broadcast_ss(r0 + 2);
                    __m256 _r03 = _mm256_broadcast_ss(r0 + 3);
                    __m256 _r04 = _mm256_broadcast_ss(r0 + 4);
                    __m256 _r05 = _mm256_broadcast_ss(r0 + 5);
                    __m256 _r06 = _mm256_broadcast_ss(r0 + 6);
                    __m256 _r07 = _mm256_broadcast_ss(r0 + 7);
                    r0 += 8;

                    __m256 _k00 = _mm256_loadu_ps(kptr);
                    __m256 _k01 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k02 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k03 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k03, _r03, _sum0);

                    __m256 _k04 = _mm256_loadu_ps(kptr);
                    __m256 _k05 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k06 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k07 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k04, _r04, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k05, _r05, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k06, _r06, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k07, _r07, _sum0);

                    //========================================

                    _r00 = _mm256_broadcast_ss(r0);
                    _r01 = _mm256_broadcast_ss(r0 + 1);
                    _r02 = _mm256_broadcast_ss(r0 + 2);
                    _r03 = _mm256_broadcast_ss(r0 + 3);
                    _r04 = _mm256_broadcast_ss(r0 + 4);
                    _r05 = _mm256_broadcast_ss(r0 + 5);
                    _r06 = _mm256_broadcast_ss(r0 + 6);
                    _r07 = _mm256_broadcast_ss(r0 + 7);
                    r0 += 8;

                    _sum1 = _mm256_fmadd_ps(_k00, _r00, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k01, _r01, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k02, _r02, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k03, _r03, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k04, _r04, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k05, _r05, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k06, _r06, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k07, _r07, _sum1);

                    _k00 = _mm256_loadu_ps(kptr);
                    _k01 = _mm256_loadu_ps(kptr + 8);
                    _k02 = _mm256_loadu_ps(kptr + 16);
                    _k03 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k03, _r03, _sum0);

                    _k04 = _mm256_loadu_ps(kptr);
                    _k05 = _mm256_loadu_ps(kptr + 8);
                    _k06 = _mm256_loadu_ps(kptr + 16);
                    _k07 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k04, _r04, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k05, _r05, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k06, _r06, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k07, _r07, _sum0);

                    _r00 = _mm256_broadcast_ss(r0);
                    _r01 = _mm256_broadcast_ss(r0 + 1);
                    _r02 = _mm256_broadcast_ss(r0 + 2);
                    _r03 = _mm256_broadcast_ss(r0 + 3);
                    _r04 = _mm256_broadcast_ss(r0 + 4);
                    _r05 = _mm256_broadcast_ss(r0 + 5);
                    _r06 = _mm256_broadcast_ss(r0 + 6);
                    _r07 = _mm256_broadcast_ss(r0 + 7);

                    _sum1 = _mm256_fmadd_ps(_k00, _r00, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k01, _r01, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k02, _r02, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k03, _r03, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k04, _r04, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k05, _r05, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k06, _r06, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k07, _r07, _sum1);
                    //===============

                    __m256 _r10 = _mm256_broadcast_ss(r1);
                    __m256 _r11 = _mm256_broadcast_ss(r1 + 1);
                    __m256 _r12 = _mm256_broadcast_ss(r1 + 2);
                    __m256 _r13 = _mm256_broadcast_ss(r1 + 3);
                    __m256 _r14 = _mm256_broadcast_ss(r1 + 4);
                    __m256 _r15 = _mm256_broadcast_ss(r1 + 5);
                    __m256 _r16 = _mm256_broadcast_ss(r1 + 6);
                    __m256 _r17 = _mm256_broadcast_ss(r1 + 7);

                    __m256 _k10 = _mm256_loadu_ps(kptr);
                    __m256 _k11 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k12 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k13 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k13, _r13, _sum0);

                    __m256 _k14 = _mm256_loadu_ps(kptr);
                    __m256 _k15 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k16 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k17 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k14, _r14, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k15, _r15, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k16, _r16, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k17, _r17, _sum0);

                    //=======================================
                    r1 += 8;
                    _r10 = _mm256_broadcast_ss(r1);
                    _r11 = _mm256_broadcast_ss(r1 + 1);
                    _r12 = _mm256_broadcast_ss(r1 + 2);
                    _r13 = _mm256_broadcast_ss(r1 + 3);
                    _r14 = _mm256_broadcast_ss(r1 + 4);
                    _r15 = _mm256_broadcast_ss(r1 + 5);
                    _r16 = _mm256_broadcast_ss(r1 + 6);
                    _r17 = _mm256_broadcast_ss(r1 + 7);

                    _sum1 = _mm256_fmadd_ps(_k10, _r10, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k11, _r11, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k12, _r12, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k13, _r13, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k14, _r14, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k15, _r15, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k16, _r16, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k17, _r17, _sum1);

                    _k10 = _mm256_loadu_ps(kptr);
                    _k11 = _mm256_loadu_ps(kptr + 8);
                    _k12 = _mm256_loadu_ps(kptr + 16);
                    _k13 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k13, _r13, _sum0);

                    _k14 = _mm256_loadu_ps(kptr);
                    _k15 = _mm256_loadu_ps(kptr + 8);
                    _k16 = _mm256_loadu_ps(kptr + 16);
                    _k17 = _mm256_loadu_ps(kptr + 24);
                    _sum0 = _mm256_fmadd_ps(_k14, _r14, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k15, _r15, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k16, _r16, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k17, _r17, _sum0);

                    r1 += 8;
                    _r10 = _mm256_broadcast_ss(r1);
                    _r11 = _mm256_broadcast_ss(r1 + 1);
                    _r12 = _mm256_broadcast_ss(r1 + 2);
                    _r13 = _mm256_broadcast_ss(r1 + 3);
                    _r14 = _mm256_broadcast_ss(r1 + 4);
                    _r15 = _mm256_broadcast_ss(r1 + 5);
                    _r16 = _mm256_broadcast_ss(r1 + 6);
                    _r17 = _mm256_broadcast_ss(r1 + 7);

                    _sum1 = _mm256_fmadd_ps(_k10, _r10, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k11, _r11, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k12, _r12, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k13, _r13, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k14, _r14, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k15, _r15, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k16, _r16, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k17, _r17, _sum1);

                    kptr -= 224;
                    _mm256_storeu_ps(outptr0, _sum0);
                    _mm256_storeu_ps(outptr0 + 8, _sum1);
                    outptr0 += 16;
                }

                for (; j < outw; j++)
                {
                    __m256 _sum = _mm256_loadu_ps(outptr0);

                    __m256 _r00 = _mm256_broadcast_ss(r0);
                    __m256 _r01 = _mm256_broadcast_ss(r0 + 1);
                    __m256 _r02 = _mm256_broadcast_ss(r0 + 2);
                    __m256 _r03 = _mm256_broadcast_ss(r0 + 3);
                    __m256 _r04 = _mm256_broadcast_ss(r0 + 4);
                    __m256 _r05 = _mm256_broadcast_ss(r0 + 5);
                    __m256 _r06 = _mm256_broadcast_ss(r0 + 6);
                    __m256 _r07 = _mm256_broadcast_ss(r0 + 7);

                    __m256 _k00 = _mm256_loadu_ps(kptr);
                    __m256 _k01 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k02 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k03 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k00, _r00, _sum);
                    _sum = _mm256_fmadd_ps(_k01, _r01, _sum);
                    _sum = _mm256_fmadd_ps(_k02, _r02, _sum);
                    _sum = _mm256_fmadd_ps(_k03, _r03, _sum);

                    __m256 _k04 = _mm256_loadu_ps(kptr);
                    __m256 _k05 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k06 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k07 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k04, _r04, _sum);
                    _sum = _mm256_fmadd_ps(_k05, _r05, _sum);
                    _sum = _mm256_fmadd_ps(_k06, _r06, _sum);
                    _sum = _mm256_fmadd_ps(_k07, _r07, _sum);

                    //========================================
                    r0 += 8;
                    _r00 = _mm256_broadcast_ss(r0);
                    _r01 = _mm256_broadcast_ss(r0 + 1);
                    _r02 = _mm256_broadcast_ss(r0 + 2);
                    _r03 = _mm256_broadcast_ss(r0 + 3);
                    _r04 = _mm256_broadcast_ss(r0 + 4);
                    _r05 = _mm256_broadcast_ss(r0 + 5);
                    _r06 = _mm256_broadcast_ss(r0 + 6);
                    _r07 = _mm256_broadcast_ss(r0 + 7);

                    _k00 = _mm256_loadu_ps(kptr);
                    _k01 = _mm256_loadu_ps(kptr + 8);
                    _k02 = _mm256_loadu_ps(kptr + 16);
                    _k03 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k00, _r00, _sum);
                    _sum = _mm256_fmadd_ps(_k01, _r01, _sum);
                    _sum = _mm256_fmadd_ps(_k02, _r02, _sum);
                    _sum = _mm256_fmadd_ps(_k03, _r03, _sum);

                    _k04 = _mm256_loadu_ps(kptr);
                    _k05 = _mm256_loadu_ps(kptr + 8);
                    _k06 = _mm256_loadu_ps(kptr + 16);
                    _k07 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k04, _r04, _sum);
                    _sum = _mm256_fmadd_ps(_k05, _r05, _sum);
                    _sum = _mm256_fmadd_ps(_k06, _r06, _sum);
                    _sum = _mm256_fmadd_ps(_k07, _r07, _sum);
                    //===============

                    __m256 _r10 = _mm256_broadcast_ss(r1);
                    __m256 _r11 = _mm256_broadcast_ss(r1 + 1);
                    __m256 _r12 = _mm256_broadcast_ss(r1 + 2);
                    __m256 _r13 = _mm256_broadcast_ss(r1 + 3);
                    __m256 _r14 = _mm256_broadcast_ss(r1 + 4);
                    __m256 _r15 = _mm256_broadcast_ss(r1 + 5);
                    __m256 _r16 = _mm256_broadcast_ss(r1 + 6);
                    __m256 _r17 = _mm256_broadcast_ss(r1 + 7);

                    __m256 _k10 = _mm256_loadu_ps(kptr);
                    __m256 _k11 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k12 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k13 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k10, _r10, _sum);
                    _sum = _mm256_fmadd_ps(_k11, _r11, _sum);
                    _sum = _mm256_fmadd_ps(_k12, _r12, _sum);
                    _sum = _mm256_fmadd_ps(_k13, _r13, _sum);

                    __m256 _k14 = _mm256_loadu_ps(kptr);
                    __m256 _k15 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k16 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k17 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k14, _r14, _sum);
                    _sum = _mm256_fmadd_ps(_k15, _r15, _sum);
                    _sum = _mm256_fmadd_ps(_k16, _r16, _sum);
                    _sum = _mm256_fmadd_ps(_k17, _r17, _sum);

                    //=======================================
                    r1 += 8;
                    _r10 = _mm256_broadcast_ss(r1);
                    _r11 = _mm256_broadcast_ss(r1 + 1);
                    _r12 = _mm256_broadcast_ss(r1 + 2);
                    _r13 = _mm256_broadcast_ss(r1 + 3);
                    _r14 = _mm256_broadcast_ss(r1 + 4);
                    _r15 = _mm256_broadcast_ss(r1 + 5);
                    _r16 = _mm256_broadcast_ss(r1 + 6);
                    _r17 = _mm256_broadcast_ss(r1 + 7);

                    _k10 = _mm256_loadu_ps(kptr);
                    _k11 = _mm256_loadu_ps(kptr + 8);
                    _k12 = _mm256_loadu_ps(kptr + 16);
                    _k13 = _mm256_loadu_ps(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k10, _r10, _sum);
                    _sum = _mm256_fmadd_ps(_k11, _r11, _sum);
                    _sum = _mm256_fmadd_ps(_k12, _r12, _sum);
                    _sum = _mm256_fmadd_ps(_k13, _r13, _sum);

                    _k14 = _mm256_loadu_ps(kptr);
                    _k15 = _mm256_loadu_ps(kptr + 8);
                    _k16 = _mm256_loadu_ps(kptr + 16);
                    _k17 = _mm256_loadu_ps(kptr + 24);
                    _sum = _mm256_fmadd_ps(_k14, _r14, _sum);
                    _sum = _mm256_fmadd_ps(_k15, _r15, _sum);
                    _sum = _mm256_fmadd_ps(_k16, _r16, _sum);
                    _sum = _mm256_fmadd_ps(_k17, _r17, _sum);

                    kptr -= 224;
                    _mm256_storeu_ps(outptr0, _sum);
                    outptr0 += 8;
                }

                r0 += 8;
                r1 += 8;
            }
        }
    }
}

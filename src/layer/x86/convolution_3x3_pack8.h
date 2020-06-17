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

static void conv3x3s1_pack8_aavx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
            const float* r2 = img0.row(2);

            const float* kptr = (const float*)kernel.channel(p).row(q);
             __m256 _k00 = _mm256_loadu_ps(kptr);
            __m256 _k01 = _mm256_loadu_ps(kptr+8);
              __m256 _k02 = _mm256_loadu_ps(kptr+16);
              __m256 _k10 = _mm256_loadu_ps(kptr+24);
              __m256 _k11 = _mm256_loadu_ps(kptr+32);
              __m256 _k12 = _mm256_loadu_ps(kptr+40);
              __m256 _k20 = _mm256_loadu_ps(kptr+48);
              __m256 _k21 = _mm256_loadu_ps(kptr+56);
            __m256 _k22 = _mm256_loadu_ps(kptr+64);
            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j < outw; j++)
                {
                    __m256 _sum0 = _mm256_loadu_ps(outptr0);
                    __m256 _r00 = _mm256_set1_ps(r0[0]);
                    __m256 _r01 = _mm256_set1_ps(r0[1]);
                    __m256 _r02 = _mm256_set1_ps(r0[2]);
                    __m256 _r03 = _mm256_set1_ps(r0[3]);
                    __m256 _r04 = _mm256_set1_ps(r0[4]);
                    __m256 _r05 = _mm256_set1_ps(r0[5]);
                    __m256 _r06 = _mm256_set1_ps(r0[6]);
                    __m256 _r07 = _mm256_set1_ps(r0[7]);
                    __m256 _r10 = _mm256_set1_ps(r1[0]);
                    __m256 _r11 = _mm256_set1_ps(r1[1]);
                    __m256 _r12 = _mm256_set1_ps(r1[2]);
                    __m256 _r13 = _mm256_set1_ps(r1[3]);
                    __m256 _r14 = _mm256_set1_ps(r1[4]);
                    __m256 _r15 = _mm256_set1_ps(r1[5]);
                    __m256 _r16 = _mm256_set1_ps(r1[6]);
                    __m256 _r17 = _mm256_set1_ps(r1[7]);
                    __m256 _r20 = _mm256_set1_ps(r1[0]);
                    __m256 _r21 = _mm256_set1_ps(r1[1]);
                    __m256 _r22 = _mm256_set1_ps(r1[2]);
                    __m256 _r23 = _mm256_set1_ps(r1[3]);
                    __m256 _r24 = _mm256_set1_ps(r1[4]);
                    __m256 _r25 = _mm256_set1_ps(r1[5]);
                    __m256 _r26 = _mm256_set1_ps(r1[6]);
                    __m256 _r27 = _mm256_set1_ps(r1[7]);
                    
                    __m256 _k00 = _mm256_loadu_ps(kptr);
                    __m256 _k01 = _mm256_loadu_ps(kptr+8);
                    __m256 _k02 = _mm256_loadu_ps(kptr+16);
                    __m256 _k03 = _mm256_loadu_ps(kptr+24);
                    __m256 _k04 = _mm256_loadu_ps(kptr+32);
                    __m256 _k05 = _mm256_loadu_ps(kptr+40);
                    __m256 _k06 = _mm256_loadu_ps(kptr+48);
                    __m256 _k07 = _mm256_loadu_ps(kptr+54);
                    __m256 _sum00 = _mm256_mul_ps(_k00,_r00);
                    __m256 _sum01 = _mm256_mul_ps(_k01,_r01);
                    __m256 _sum02 = _mm256_mul_ps(_k02,_r02);
                    __m256 _sum03 = _mm256_mul_ps(_k03,_r03);
                    __m256 _sum04 = _mm256_mul_ps(_k04,_r04);
                    __m256 _sum05 = _mm256_mul_ps(_k05,_r05);
                    __m256 _sum06 = _mm256_mul_ps(_k06,_r06);
                    __m256 _sum07 = _mm256_mul_ps(_k07,_r07);
                    
                  
                }

                r0 += 4 * 4;
                r1 += 4 * 4;
                r2 += 4 * 4;
            }
        }
    }
}

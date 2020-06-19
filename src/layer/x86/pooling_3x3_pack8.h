
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

static void pooling3x3s2_max_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 8;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 8);
                __m256 _r02 = _mm256_loadu_ps(r0 + 16);
                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 8);
                __m256 _r12 = _mm256_loadu_ps(r1 + 16);
                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 8);
                __m256 _r22 = _mm256_loadu_ps(r2 + 16);

                __m256 _max00 = _mm256_max_ps(_r00, _r01);
                _max00 = _mm256_max_ps(_max00, _r02);
                _max00 = _mm256_max_ps(_max00, _r10);
                _max00 = _mm256_max_ps(_max00, _r11);
                __m256 _max01 = _mm256_max_ps(_r12, _r20);
                _max01 = _mm256_max_ps(_max01, _r21);
                _max01 = _mm256_max_ps(_max01, _r22);

                __m256 _r03 = _mm256_loadu_ps(r0 + 24);
                __m256 _r04 = _mm256_loadu_ps(r0 + 32);
                __m256 _r13 = _mm256_loadu_ps(r1 + 24);
                __m256 _r14 = _mm256_loadu_ps(r1 + 32);
                __m256 _r23 = _mm256_loadu_ps(r2 + 24);
                __m256 _r24 = _mm256_loadu_ps(r2 + 32);

                _mm256_storeu_ps(outptr, _mm256_max_ps(_max00, _max01));

                __m256 _max10 = _mm256_max_ps(_r03, _r04);
                _max10 = _mm256_max_ps(_max10, _r02);
                _max10 = _mm256_max_ps(_max10, _r13);
                _max10 = _mm256_max_ps(_max10, _r14);
                __m256 _max11 = _mm256_max_ps(_r12, _r23);
                _max10 = _mm256_max_ps(_max10, _r24);
                _max10 = _mm256_max_ps(_max10, _r22);

                __m256 _r05 = _mm256_loadu_ps(r0 + 40);
                __m256 _r06 = _mm256_loadu_ps(r0 + 48);
                __m256 _r15 = _mm256_loadu_ps(r1 + 40);
                __m256 _r16 = _mm256_loadu_ps(r1 + 48);
                __m256 _r25 = _mm256_loadu_ps(r2 + 40);
                __m256 _r26 = _mm256_loadu_ps(r2 + 48);

                _mm256_storeu_ps(outptr + 8, _mm256_max_ps(_max10, _max11));

                __m256 _max20 = _mm256_max_ps(_r05, _r06);
                _max20 = _mm256_max_ps(_max20, _r04);
                _max20 = _mm256_max_ps(_max20, _r15);
                _max20 = _mm256_max_ps(_max20, _r16);
                __m256 _max21 = _mm256_max_ps(_r14, _r25);
                _max20 = _mm256_max_ps(_max20, _r26);
                _max20 = _mm256_max_ps(_max20, _r24);

                __m256 _r07 = _mm256_loadu_ps(r0 + 56);
                __m256 _r08 = _mm256_loadu_ps(r0 + 64);
                __m256 _r17 = _mm256_loadu_ps(r1 + 56);
                __m256 _r18 = _mm256_loadu_ps(r1 + 64);
                __m256 _r27 = _mm256_loadu_ps(r2 + 56);
                __m256 _r28 = _mm256_loadu_ps(r2 + 64);

                _mm256_storeu_ps(outptr + 16, _mm256_max_ps(_max20, _max21));

                __m256 _max30 = _mm256_max_ps(_r07, _r08);
                _max30 = _mm256_max_ps(_max30, _r06);
                _max30 = _mm256_max_ps(_max30, _r17);
                _max30 = _mm256_max_ps(_max30, _r18);
                __m256 _max31 = _mm256_max_ps(_r16, _r27);
                _max30 = _mm256_max_ps(_max30, _r28);
                _max30 = _mm256_max_ps(_max30, _r26);

                _mm256_storeu_ps(outptr + 24, _mm256_max_ps(_max30, _max31));

                r0 += 64;
                r1 += 64;
                r2 += 64;
                outptr += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 8);
                __m256 _r02 = _mm256_loadu_ps(r0 + 16);
                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 8);
                __m256 _r12 = _mm256_loadu_ps(r1 + 16);
                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 8);
                __m256 _r22 = _mm256_loadu_ps(r2 + 16);

                __m256 _max00 = _mm256_max_ps(_r00, _r01);
                _max00 = _mm256_max_ps(_max00, _r02);
                _max00 = _mm256_max_ps(_max00, _r10);
                _max00 = _mm256_max_ps(_max00, _r11);
                __m256 _max01 = _mm256_max_ps(_r12, _r20);
                _max01 = _mm256_max_ps(_max01, _r21);
                _max01 = _mm256_max_ps(_max01, _r22);

                __m256 _r03 = _mm256_loadu_ps(r0 + 24);
                __m256 _r04 = _mm256_loadu_ps(r0 + 32);
                __m256 _r13 = _mm256_loadu_ps(r1 + 24);
                __m256 _r14 = _mm256_loadu_ps(r1 + 32);
                __m256 _r23 = _mm256_loadu_ps(r2 + 24);
                __m256 _r24 = _mm256_loadu_ps(r2 + 32);

                _mm256_storeu_ps(outptr, _mm256_max_ps(_max00, _max01));

                __m256 _max10 = _mm256_max_ps(_r03, _r04);
                _max10 = _mm256_max_ps(_max10, _r02);
                _max10 = _mm256_max_ps(_max10, _r13);
                _max10 = _mm256_max_ps(_max10, _r14);
                __m256 _max11 = _mm256_max_ps(_r12, _r23);
                _max10 = _mm256_max_ps(_max10, _r24);
                _max10 = _mm256_max_ps(_max10, _r22);

                _mm256_storeu_ps(outptr + 8, _mm256_max_ps(_max10, _max11));

                r0 += 32;
                r1 += 32;
                r2 += 32;
                outptr += 16;
            }

            for (; j < outw; j++)
            {
                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 8);
                __m256 _r02 = _mm256_loadu_ps(r0 + 16);
                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 8);
                __m256 _r12 = _mm256_loadu_ps(r1 + 16);
                __m256 _r20 = _mm256_loadu_ps(r2);
                __m256 _r21 = _mm256_loadu_ps(r2 + 8);
                __m256 _r22 = _mm256_loadu_ps(r2 + 16);

                __m256 _max0 = _mm256_max_ps(_r00, _r01);
                _max0 = _mm256_max_ps(_max0, _r02);
                _max0 = _mm256_max_ps(_max0, _r10);
                _max0 = _mm256_max_ps(_max0, _r11);
                __m256 _max1 = _mm256_max_ps(_r12, _r20);
                _max1 = _mm256_max_ps(_max1, _r21);
                _max1 = _mm256_max_ps(_max1, _r22);

                _mm256_storeu_ps(outptr, _mm256_max_ps(_max0, _max1));

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr += 8;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

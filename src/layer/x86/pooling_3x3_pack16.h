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

static void pooling3x3s2_max_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 16;

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
                __m512 _r00 = _mm512_loadu_ps(r0);
                __m512 _r01 = _mm512_loadu_ps(r0 + 16);
                __m512 _r02 = _mm512_loadu_ps(r0 + 32);
                __m512 _r10 = _mm512_loadu_ps(r1);
                __m512 _r11 = _mm512_loadu_ps(r1 + 16);
                __m512 _r12 = _mm512_loadu_ps(r1 + 32);
                __m512 _r20 = _mm512_loadu_ps(r2);
                __m512 _r21 = _mm512_loadu_ps(r2 + 16);
                __m512 _r22 = _mm512_loadu_ps(r2 + 32);

                __m512 _max00 = _mm512_max_ps(_r00, _r01);
                _max00 = _mm512_max_ps(_max00, _r02);
                _max00 = _mm512_max_ps(_max00, _r10);
                _max00 = _mm512_max_ps(_max00, _r11);
                __m512 _max01 = _mm512_max_ps(_r12, _r20);
                _max01 = _mm512_max_ps(_max01, _r21);
                _max01 = _mm512_max_ps(_max01, _r22);

                __m512 _r03 = _mm512_loadu_ps(r0 + 48);
                __m512 _r04 = _mm512_loadu_ps(r0 + 64);
                __m512 _r13 = _mm512_loadu_ps(r1 + 48);
                __m512 _r14 = _mm512_loadu_ps(r1 + 64);
                __m512 _r23 = _mm512_loadu_ps(r2 + 48);
                __m512 _r24 = _mm512_loadu_ps(r2 + 64);

                _mm512_storeu_ps(outptr, _mm512_max_ps(_max00, _max01));

                __m512 _max10 = _mm512_max_ps(_r03, _r04);
                _max10 = _mm512_max_ps(_max10, _r02);
                _max10 = _mm512_max_ps(_max10, _r13);
                _max10 = _mm512_max_ps(_max10, _r14);
                __m512 _max11 = _mm512_max_ps(_r12, _r23);
                _max10 = _mm512_max_ps(_max10, _r24);
                _max10 = _mm512_max_ps(_max10, _r22);

                __m512 _r05 = _mm512_loadu_ps(r0 + 80);
                __m512 _r06 = _mm512_loadu_ps(r0 + 96);
                __m512 _r15 = _mm512_loadu_ps(r1 + 80);
                __m512 _r16 = _mm512_loadu_ps(r1 + 96);
                __m512 _r25 = _mm512_loadu_ps(r2 + 80);
                __m512 _r26 = _mm512_loadu_ps(r2 + 96);

                _mm512_storeu_ps(outptr + 16, _mm512_max_ps(_max10, _max11));

                __m512 _max20 = _mm512_max_ps(_r05, _r06);
                _max20 = _mm512_max_ps(_max20, _r04);
                _max20 = _mm512_max_ps(_max20, _r15);
                _max20 = _mm512_max_ps(_max20, _r16);
                __m512 _max21 = _mm512_max_ps(_r14, _r25);
                _max20 = _mm512_max_ps(_max20, _r26);
                _max20 = _mm512_max_ps(_max20, _r24);

                __m512 _r07 = _mm512_loadu_ps(r0 + 112);
                __m512 _r08 = _mm512_loadu_ps(r0 + 128);
                __m512 _r17 = _mm512_loadu_ps(r1 + 112);
                __m512 _r18 = _mm512_loadu_ps(r1 + 128);
                __m512 _r27 = _mm512_loadu_ps(r2 + 112);
                __m512 _r28 = _mm512_loadu_ps(r2 + 128);

                _mm512_storeu_ps(outptr + 32, _mm512_max_ps(_max20, _max21));

                __m512 _max30 = _mm512_max_ps(_r07, _r08);
                _max30 = _mm512_max_ps(_max30, _r06);
                _max30 = _mm512_max_ps(_max30, _r17);
                _max30 = _mm512_max_ps(_max30, _r18);
                __m512 _max31 = _mm512_max_ps(_r16, _r27);
                _max30 = _mm512_max_ps(_max30, _r28);
                _max30 = _mm512_max_ps(_max30, _r26);

                _mm512_storeu_ps(outptr + 48, _mm512_max_ps(_max30, _max31));

                r0 += 128;
                r1 += 128;
                r2 += 128;
                outptr += 64;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m512 _r00 = _mm512_loadu_ps(r0);
                __m512 _r01 = _mm512_loadu_ps(r0 + 16);
                __m512 _r02 = _mm512_loadu_ps(r0 + 32);
                __m512 _r10 = _mm512_loadu_ps(r1);
                __m512 _r11 = _mm512_loadu_ps(r1 + 16);
                __m512 _r12 = _mm512_loadu_ps(r1 + 32);
                __m512 _r20 = _mm512_loadu_ps(r2);
                __m512 _r21 = _mm512_loadu_ps(r2 + 16);
                __m512 _r22 = _mm512_loadu_ps(r2 + 32);

                __m512 _max00 = _mm512_max_ps(_r00, _r01);
                _max00 = _mm512_max_ps(_max00, _r02);
                _max00 = _mm512_max_ps(_max00, _r10);
                _max00 = _mm512_max_ps(_max00, _r11);
                __m512 _max01 = _mm512_max_ps(_r12, _r20);
                _max01 = _mm512_max_ps(_max01, _r21);
                _max01 = _mm512_max_ps(_max01, _r22);

                __m512 _r03 = _mm512_loadu_ps(r0 + 48);
                __m512 _r04 = _mm512_loadu_ps(r0 + 64);
                __m512 _r13 = _mm512_loadu_ps(r1 + 48);
                __m512 _r14 = _mm512_loadu_ps(r1 + 64);
                __m512 _r23 = _mm512_loadu_ps(r2 + 48);
                __m512 _r24 = _mm512_loadu_ps(r2 + 64);

                _mm512_storeu_ps(outptr, _mm512_max_ps(_max00, _max01));

                __m512 _max10 = _mm512_max_ps(_r03, _r04);
                _max10 = _mm512_max_ps(_max10, _r02);
                _max10 = _mm512_max_ps(_max10, _r13);
                _max10 = _mm512_max_ps(_max10, _r14);
                __m512 _max11 = _mm512_max_ps(_r12, _r23);
                _max10 = _mm512_max_ps(_max10, _r24);
                _max10 = _mm512_max_ps(_max10, _r22);

                _mm512_storeu_ps(outptr + 16, _mm512_max_ps(_max10, _max11));

                r0 += 64;
                r1 += 64;
                r2 += 64;
                outptr += 32;
            }

            for (; j < outw; j++)
            {
                __m512 _r00 = _mm512_loadu_ps(r0);
                __m512 _r01 = _mm512_loadu_ps(r0 + 16);
                __m512 _r02 = _mm512_loadu_ps(r0 + 32);
                __m512 _r10 = _mm512_loadu_ps(r1);
                __m512 _r11 = _mm512_loadu_ps(r1 + 16);
                __m512 _r12 = _mm512_loadu_ps(r1 + 32);
                __m512 _r20 = _mm512_loadu_ps(r2);
                __m512 _r21 = _mm512_loadu_ps(r2 + 16);
                __m512 _r22 = _mm512_loadu_ps(r2 + 32);

                __m512 _max0 = _mm512_max_ps(_r00, _r01);
                _max0 = _mm512_max_ps(_max0, _r02);
                _max0 = _mm512_max_ps(_max0, _r10);
                _max0 = _mm512_max_ps(_max0, _r11);
                __m512 _max1 = _mm512_max_ps(_r12, _r20);
                _max1 = _mm512_max_ps(_max1, _r21);
                _max1 = _mm512_max_ps(_max1, _r22);

                _mm512_storeu_ps(outptr, _mm512_max_ps(_max0, _max1));

                r0 += 32;
                r1 += 32;
                r2 += 32;
                outptr += 16;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

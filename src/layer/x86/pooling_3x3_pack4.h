// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pooling3x3s2_max_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 4;

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
            for (; j + 1 < outw; j += 2)
            {
                __m128 _r00 = _mm_loadu_ps(r0);
                __m128 _r01 = _mm_loadu_ps(r0 + 4);
                __m128 _r02 = _mm_loadu_ps(r0 + 8);
                __m128 _r10 = _mm_loadu_ps(r1);
                __m128 _r11 = _mm_loadu_ps(r1 + 4);
                __m128 _r12 = _mm_loadu_ps(r1 + 8);
                __m128 _r20 = _mm_loadu_ps(r2);
                __m128 _r21 = _mm_loadu_ps(r2 + 4);
                __m128 _r22 = _mm_loadu_ps(r2 + 8);

                __m128 _max00 = _mm_max_ps(_r00, _r01);
                _max00 = _mm_max_ps(_max00, _r02);
                _max00 = _mm_max_ps(_max00, _r10);
                _max00 = _mm_max_ps(_max00, _r11);
                __m128 _max01 = _mm_max_ps(_r12, _r20);
                _max01 = _mm_max_ps(_max01, _r21);
                _max01 = _mm_max_ps(_max01, _r22);

                __m128 _r03 = _mm_loadu_ps(r0 + 12);
                __m128 _r04 = _mm_loadu_ps(r0 + 16);
                __m128 _r13 = _mm_loadu_ps(r1 + 12);
                __m128 _r14 = _mm_loadu_ps(r1 + 16);
                __m128 _r23 = _mm_loadu_ps(r2 + 12);
                __m128 _r24 = _mm_loadu_ps(r2 + 16);

                _mm_storeu_ps(outptr, _mm_max_ps(_max00, _max01));

                __m128 _max10 = _mm_max_ps(_r03, _r04);
                _max10 = _mm_max_ps(_max10, _r02);
                _max10 = _mm_max_ps(_max10, _r13);
                _max10 = _mm_max_ps(_max10, _r14);
                __m128 _max11 = _mm_max_ps(_r12, _r23);
                _max10 = _mm_max_ps(_max10, _r24);
                _max10 = _mm_max_ps(_max10, _r22);

                _mm_storeu_ps(outptr + 4, _mm_max_ps(_max10, _max11));

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr += 8;
            }

            for (; j < outw; j++)
            {
                __m128 _r00 = _mm_loadu_ps(r0);
                __m128 _r01 = _mm_loadu_ps(r0 + 4);
                __m128 _r02 = _mm_loadu_ps(r0 + 8);
                __m128 _r10 = _mm_loadu_ps(r1);
                __m128 _r11 = _mm_loadu_ps(r1 + 4);
                __m128 _r12 = _mm_loadu_ps(r1 + 8);
                __m128 _r20 = _mm_loadu_ps(r2);
                __m128 _r21 = _mm_loadu_ps(r2 + 4);
                __m128 _r22 = _mm_loadu_ps(r2 + 8);

                __m128 _max0 = _mm_max_ps(_r00, _r01);
                _max0 = _mm_max_ps(_max0, _r02);
                _max0 = _mm_max_ps(_max0, _r10);
                _max0 = _mm_max_ps(_max0, _r11);
                __m128 _max1 = _mm_max_ps(_r12, _r20);
                _max1 = _mm_max_ps(_max1, _r21);
                _max1 = _mm_max_ps(_max1, _r22);

                _mm_storeu_ps(outptr, _mm_max_ps(_max0, _max1));

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

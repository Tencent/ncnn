// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pooling2x2s2_max_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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

        for (int i = 0; i < outh; i++)
        {
            int j = 0;

            for (; j < outw; j++)
            {
                __m256 _r00 = _mm256_loadu_ps(r0);
                __m256 _r01 = _mm256_loadu_ps(r0 + 8);
                __m256 _r10 = _mm256_loadu_ps(r1);
                __m256 _r11 = _mm256_loadu_ps(r1 + 8);

                __m256 _max0 = _mm256_max_ps(_r00, _r01);
                __m256 _max1 = _mm256_max_ps(_r10, _r11);
                __m256 _max = _mm256_max_ps(_max0, _max1);

                _mm256_storeu_ps(outptr, _max);

                r0 += 16;
                r1 += 16;
                outptr += 8;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}

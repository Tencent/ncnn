// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pooling3x3s2_max_pack4_lsx(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 8, 0);
                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 8, 0);
                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 8, 0);

                __m128 _max00 = __lsx_vfmax_s(_r00, _r01);
                _max00 = __lsx_vfmax_s(_max00, _r02);
                _max00 = __lsx_vfmax_s(_max00, _r10);
                _max00 = __lsx_vfmax_s(_max00, _r11);
                __m128 _max01 = __lsx_vfmax_s(_r12, _r20);
                _max01 = __lsx_vfmax_s(_max01, _r21);
                _max01 = __lsx_vfmax_s(_max01, _r22);

                __m128 _r03 = (__m128)__lsx_vld(r0 + 12, 0);
                __m128 _r04 = (__m128)__lsx_vld(r0 + 16, 0);
                __m128 _r13 = (__m128)__lsx_vld(r1 + 12, 0);
                __m128 _r14 = (__m128)__lsx_vld(r1 + 16, 0);
                __m128 _r23 = (__m128)__lsx_vld(r2 + 12, 0);
                __m128 _r24 = (__m128)__lsx_vld(r2 + 16, 0);

                __lsx_vst(__lsx_vfmax_s(_max00, _max01), outptr, 0);

                __m128 _max10 = __lsx_vfmax_s(_r03, _r04);
                _max10 = __lsx_vfmax_s(_max10, _r02);
                _max10 = __lsx_vfmax_s(_max10, _r13);
                _max10 = __lsx_vfmax_s(_max10, _r14);
                __m128 _max11 = __lsx_vfmax_s(_r12, _r23);
                _max10 = __lsx_vfmax_s(_max10, _r24);
                _max10 = __lsx_vfmax_s(_max10, _r22);

                __lsx_vst(__lsx_vfmax_s(_max10, _max11), outptr + 4, 0);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr += 8;
            }

            for (; j < outw; j++)
            {
                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r02 = (__m128)__lsx_vld(r0 + 8, 0);
                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r11 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r12 = (__m128)__lsx_vld(r1 + 8, 0);
                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r21 = (__m128)__lsx_vld(r2 + 4, 0);
                __m128 _r22 = (__m128)__lsx_vld(r2 + 8, 0);

                __m128 _max0 = __lsx_vfmax_s(_r00, _r01);
                _max0 = __lsx_vfmax_s(_max0, _r02);
                _max0 = __lsx_vfmax_s(_max0, _r10);
                _max0 = __lsx_vfmax_s(_max0, _r11);
                __m128 _max1 = __lsx_vfmax_s(_r12, _r20);
                _max1 = __lsx_vfmax_s(_max1, _r21);
                _max1 = __lsx_vfmax_s(_max1, _r22);

                __lsx_vst(__lsx_vfmax_s(_max0, _max1), outptr, 0);

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

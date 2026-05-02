// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __loongarch_asx
static void pooling3x3s2_max_pack8_bf16s_lasx(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
        unsigned short* outptr = top_blob.channel(q);

        const unsigned short* r0 = img0.row<const unsigned short>(0);
        const unsigned short* r1 = img0.row<const unsigned short>(1);
        const unsigned short* r2 = img0.row<const unsigned short>(2);
        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m256 _r00 = bfloat2float_lasx((__m128i)__lsx_vld(r0, 0));
                __m256 _r01 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 8, 0));
                __m256 _r02 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 16, 0));
                __m256 _r10 = bfloat2float_lasx((__m128i)__lsx_vld(r1, 0));
                __m256 _r11 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 8, 0));
                __m256 _r12 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 16, 0));
                __m256 _r20 = bfloat2float_lasx((__m128i)__lsx_vld(r2, 0));
                __m256 _r21 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 8, 0));
                __m256 _r22 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 16, 0));

                __m256 _max00 = __lasx_xvfmax_s(_r00, _r01);
                _max00 = __lasx_xvfmax_s(_max00, _r02);
                _max00 = __lasx_xvfmax_s(_max00, _r10);
                _max00 = __lasx_xvfmax_s(_max00, _r11);
                __m256 _max01 = __lasx_xvfmax_s(_r12, _r20);
                _max01 = __lasx_xvfmax_s(_max01, _r21);
                _max01 = __lasx_xvfmax_s(_max01, _r22);

                __m256 _r03 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 24, 0));
                __m256 _r04 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 32, 0));
                __m256 _r13 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 24, 0));
                __m256 _r14 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 32, 0));
                __m256 _r23 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 24, 0));
                __m256 _r24 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 32, 0));

                __lsx_vst(float2bfloat_lasx(__lasx_xvfmax_s(_max00, _max01)), outptr, 0);

                __m256 _max10 = __lasx_xvfmax_s(_r03, _r04);
                _max10 = __lasx_xvfmax_s(_max10, _r02);
                _max10 = __lasx_xvfmax_s(_max10, _r13);
                _max10 = __lasx_xvfmax_s(_max10, _r14);
                __m256 _max11 = __lasx_xvfmax_s(_r12, _r23);
                _max10 = __lasx_xvfmax_s(_max10, _r24);
                _max10 = __lasx_xvfmax_s(_max10, _r22);

                __m256 _r05 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 40, 0));
                __m256 _r06 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 48, 0));
                __m256 _r15 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 40, 0));
                __m256 _r16 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 48, 0));
                __m256 _r25 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 40, 0));
                __m256 _r26 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 48, 0));

                __lsx_vst(float2bfloat_lasx(__lasx_xvfmax_s(_max10, _max11)), outptr + 8, 0);

                __m256 _max20 = __lasx_xvfmax_s(_r05, _r06);
                _max20 = __lasx_xvfmax_s(_max20, _r04);
                _max20 = __lasx_xvfmax_s(_max20, _r15);
                _max20 = __lasx_xvfmax_s(_max20, _r16);
                __m256 _max21 = __lasx_xvfmax_s(_r14, _r25);
                _max20 = __lasx_xvfmax_s(_max20, _r26);
                _max20 = __lasx_xvfmax_s(_max20, _r24);

                __m256 _r07 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 56, 0));
                __m256 _r08 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 64, 0));
                __m256 _r17 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 56, 0));
                __m256 _r18 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 64, 0));
                __m256 _r27 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 56, 0));
                __m256 _r28 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 64, 0));

                __lsx_vst(float2bfloat_lasx(__lasx_xvfmax_s(_max20, _max21)), outptr + 16, 0);

                __m256 _max30 = __lasx_xvfmax_s(_r07, _r08);
                _max30 = __lasx_xvfmax_s(_max30, _r06);
                _max30 = __lasx_xvfmax_s(_max30, _r17);
                _max30 = __lasx_xvfmax_s(_max30, _r18);
                __m256 _max31 = __lasx_xvfmax_s(_r16, _r27);
                _max30 = __lasx_xvfmax_s(_max30, _r28);
                _max30 = __lasx_xvfmax_s(_max30, _r26);

                __lsx_vst(float2bfloat_lasx(__lasx_xvfmax_s(_max30, _max31)), outptr + 24, 0);

                r0 += 64;
                r1 += 64;
                r2 += 64;
                outptr += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _r00 = bfloat2float_lasx((__m128i)__lsx_vld(r0, 0));
                __m256 _r01 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 8, 0));
                __m256 _r02 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 16, 0));
                __m256 _r10 = bfloat2float_lasx((__m128i)__lsx_vld(r1, 0));
                __m256 _r11 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 8, 0));
                __m256 _r12 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 16, 0));
                __m256 _r20 = bfloat2float_lasx((__m128i)__lsx_vld(r2, 0));
                __m256 _r21 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 8, 0));
                __m256 _r22 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 16, 0));

                __m256 _max00 = __lasx_xvfmax_s(_r00, _r01);
                _max00 = __lasx_xvfmax_s(_max00, _r02);
                _max00 = __lasx_xvfmax_s(_max00, _r10);
                _max00 = __lasx_xvfmax_s(_max00, _r11);
                __m256 _max01 = __lasx_xvfmax_s(_r12, _r20);
                _max01 = __lasx_xvfmax_s(_max01, _r21);
                _max01 = __lasx_xvfmax_s(_max01, _r22);

                __m256 _r03 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 24, 0));
                __m256 _r04 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 32, 0));
                __m256 _r13 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 24, 0));
                __m256 _r14 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 32, 0));
                __m256 _r23 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 24, 0));
                __m256 _r24 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 32, 0));

                __lsx_vst(float2bfloat_lasx(__lasx_xvfmax_s(_max00, _max01)), outptr, 0);

                __m256 _max10 = __lasx_xvfmax_s(_r03, _r04);
                _max10 = __lasx_xvfmax_s(_max10, _r02);
                _max10 = __lasx_xvfmax_s(_max10, _r13);
                _max10 = __lasx_xvfmax_s(_max10, _r14);
                __m256 _max11 = __lasx_xvfmax_s(_r12, _r23);
                _max10 = __lasx_xvfmax_s(_max10, _r24);
                _max10 = __lasx_xvfmax_s(_max10, _r22);

                __lsx_vst(float2bfloat_lasx(__lasx_xvfmax_s(_max10, _max11)), outptr + 8, 0);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                outptr += 16;
            }

            for (; j < outw; j++)
            {
                __m256 _r00 = bfloat2float_lasx((__m128i)__lsx_vld(r0, 0));
                __m256 _r01 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 8, 0));
                __m256 _r02 = bfloat2float_lasx((__m128i)__lsx_vld(r0 + 16, 0));
                __m256 _r10 = bfloat2float_lasx((__m128i)__lsx_vld(r1, 0));
                __m256 _r11 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 8, 0));
                __m256 _r12 = bfloat2float_lasx((__m128i)__lsx_vld(r1 + 16, 0));
                __m256 _r20 = bfloat2float_lasx((__m128i)__lsx_vld(r2, 0));
                __m256 _r21 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 8, 0));
                __m256 _r22 = bfloat2float_lasx((__m128i)__lsx_vld(r2 + 16, 0));

                __m256 _max0 = __lasx_xvfmax_s(_r00, _r01);
                _max0 = __lasx_xvfmax_s(_max0, _r02);
                _max0 = __lasx_xvfmax_s(_max0, _r10);
                _max0 = __lasx_xvfmax_s(_max0, _r11);
                __m256 _max1 = __lasx_xvfmax_s(_r12, _r20);
                _max1 = __lasx_xvfmax_s(_max1, _r21);
                _max1 = __lasx_xvfmax_s(_max1, _r22);

                __lsx_vst(float2bfloat_lasx(__lasx_xvfmax_s(_max0, _max1)), outptr, 0);

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
#endif // __loongarch_asx

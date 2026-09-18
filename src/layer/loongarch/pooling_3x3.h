// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pooling3x3s2_max_lsx(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const float* img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                __m128 _r04 = (__m128)__lsx_vld(r0 + 4, 0);
                __m128 _r10 = (__m128)__lsx_vld(r1, 0);
                __m128 _r14 = (__m128)__lsx_vld(r1 + 4, 0);
                __m128 _r20 = (__m128)__lsx_vld(r2, 0);
                __m128 _r24 = (__m128)__lsx_vld(r2 + 4, 0);

                __m128 _r0max0 = __lsx_vfmax_s(_r00, (__m128)__lsx_vshuf4i_w((__m128i)_r00, _LSX_SHUFFLE(2, 3, 0, 1)));
                __m128 _r0max1 = __lsx_vfmax_s(_r04, (__m128)__lsx_vshuf4i_w((__m128i)_r04, _LSX_SHUFFLE(2, 3, 0, 1)));
                __m128 _r0max = (__m128)__lsx_vpickev_w((__m128i)_r0max1, (__m128i)_r0max0);
                __m128i _r0v2 = __lsx_vshuf4i_w(__lsx_vpickev_w((__m128i)_r04, (__m128i)_r00), _LSX_SHUFFLE(0, 3, 2, 1));
                _r0v2 = __lsx_vinsgr2vr_w(_r0v2, ((const int*)r0)[8], 3);
                _r0max = __lsx_vfmax_s(_r0max, (__m128)_r0v2);

                __m128 _r1max0 = __lsx_vfmax_s(_r10, (__m128)__lsx_vshuf4i_w((__m128i)_r10, _LSX_SHUFFLE(2, 3, 0, 1)));
                __m128 _r1max1 = __lsx_vfmax_s(_r14, (__m128)__lsx_vshuf4i_w((__m128i)_r14, _LSX_SHUFFLE(2, 3, 0, 1)));
                __m128 _r1max = (__m128)__lsx_vpickev_w((__m128i)_r1max1, (__m128i)_r1max0);
                __m128i _r1v2 = __lsx_vshuf4i_w(__lsx_vpickev_w((__m128i)_r14, (__m128i)_r10), _LSX_SHUFFLE(0, 3, 2, 1));
                _r1v2 = __lsx_vinsgr2vr_w(_r1v2, ((const int*)r1)[8], 3);
                _r1max = __lsx_vfmax_s(_r1max, (__m128)_r1v2);

                __m128 _r2max0 = __lsx_vfmax_s(_r20, (__m128)__lsx_vshuf4i_w((__m128i)_r20, _LSX_SHUFFLE(2, 3, 0, 1)));
                __m128 _r2max1 = __lsx_vfmax_s(_r24, (__m128)__lsx_vshuf4i_w((__m128i)_r24, _LSX_SHUFFLE(2, 3, 0, 1)));
                __m128 _r2max = (__m128)__lsx_vpickev_w((__m128i)_r2max1, (__m128i)_r2max0);
                __m128i _r2v2 = __lsx_vshuf4i_w(__lsx_vpickev_w((__m128i)_r24, (__m128i)_r20), _LSX_SHUFFLE(0, 3, 2, 1));
                _r2v2 = __lsx_vinsgr2vr_w(_r2v2, ((const int*)r2)[8], 3);
                _r2max = __lsx_vfmax_s(_r2max, (__m128)_r2v2);

                __m128 _max = __lsx_vfmax_s(_r0max, _r1max);
                _max = __lsx_vfmax_s(_max, _r2max);
                __lsx_vst(_max, outptr, 0);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 4;
            }

            for (; j < outw; j++)
            {
                float max0 = std::max(std::max(r0[0], r0[1]), r0[2]);
                float max1 = std::max(std::max(r1[0], r1[1]), r1[2]);
                float max2 = std::max(std::max(r2[0], r2[1]), r2[2]);

                *outptr = std::max(std::max(max0, max1), max2);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

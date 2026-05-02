// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __loongarch_sx
static void pooling2x2s2_max_pack4_bf16s_lsx(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
        unsigned short* outptr = top_blob.channel(q);

        const unsigned short* r0 = img0.row<const unsigned short>(0);
        const unsigned short* r1 = img0.row<const unsigned short>(1);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;

            for (; j < outw; j++)
            {
                __m128 _r00 = bfloat2float_lsx(r0);
                __m128 _r01 = bfloat2float_lsx(r0 + 4);
                __m128 _r10 = bfloat2float_lsx(r1);
                __m128 _r11 = bfloat2float_lsx(r1 + 4);

                __m128 _max0 = __lsx_vfmax_s(_r00, _r01);
                __m128 _max1 = __lsx_vfmax_s(_r10, _r11);
                __m128 _max = __lsx_vfmax_s(_max0, _max1);

                __lsx_vstelm_d(float2bfloat_lsx(_max), outptr, 0, 0);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
#endif // __loongarch_sx

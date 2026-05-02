// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pooling2x2s2_max_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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

        for (int i = 0; i < outh; i++)
        {
            int j = 0;

            for (; j < outw; j++)
            {
                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);

                v4f32 _max0 = __msa_fmax_w(_r00, _r01);
                v4f32 _max1 = __msa_fmax_w(_r10, _r11);
                v4f32 _max = __msa_fmax_w(_max0, _max1);

                __msa_st_w((v4i32)_max, outptr, 0);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}

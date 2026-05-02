// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pooling3x3s2_max_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 8, 0);
                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 8, 0);
                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 8, 0);

                v4f32 _max00 = __msa_fmax_w(_r00, _r01);
                _max00 = __msa_fmax_w(_max00, _r02);
                _max00 = __msa_fmax_w(_max00, _r10);
                _max00 = __msa_fmax_w(_max00, _r11);
                v4f32 _max01 = __msa_fmax_w(_r12, _r20);
                _max01 = __msa_fmax_w(_max01, _r21);
                _max01 = __msa_fmax_w(_max01, _r22);

                v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 12, 0);
                v4f32 _r04 = (v4f32)__msa_ld_w(r0 + 16, 0);
                v4f32 _r13 = (v4f32)__msa_ld_w(r1 + 12, 0);
                v4f32 _r14 = (v4f32)__msa_ld_w(r1 + 16, 0);
                v4f32 _r23 = (v4f32)__msa_ld_w(r2 + 12, 0);
                v4f32 _r24 = (v4f32)__msa_ld_w(r2 + 16, 0);

                __msa_st_w((v4i32)__msa_fmax_w(_max00, _max01), outptr, 0);

                v4f32 _max10 = __msa_fmax_w(_r03, _r04);
                _max10 = __msa_fmax_w(_max10, _r02);
                _max10 = __msa_fmax_w(_max10, _r13);
                _max10 = __msa_fmax_w(_max10, _r14);
                v4f32 _max11 = __msa_fmax_w(_r12, _r23);
                _max10 = __msa_fmax_w(_max10, _r24);
                _max10 = __msa_fmax_w(_max10, _r22);

                __msa_st_w((v4i32)__msa_fmax_w(_max10, _max11), outptr + 4, 0);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr += 8;
            }

            for (; j < outw; j++)
            {
                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 8, 0);
                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r12 = (v4f32)__msa_ld_w(r1 + 8, 0);
                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r21 = (v4f32)__msa_ld_w(r2 + 4, 0);
                v4f32 _r22 = (v4f32)__msa_ld_w(r2 + 8, 0);

                v4f32 _max0 = __msa_fmax_w(_r00, _r01);
                _max0 = __msa_fmax_w(_max0, _r02);
                _max0 = __msa_fmax_w(_max0, _r10);
                _max0 = __msa_fmax_w(_max0, _r11);
                v4f32 _max1 = __msa_fmax_w(_r12, _r20);
                _max1 = __msa_fmax_w(_max1, _r21);
                _max1 = __msa_fmax_w(_max1, _r22);

                __msa_st_w((v4i32)__msa_fmax_w(_max0, _max1), outptr, 0);

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

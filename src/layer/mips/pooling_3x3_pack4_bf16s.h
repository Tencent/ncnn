// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __mips_msa
static void pooling3x3s2_max_pack4_bf16s_msa(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
        const unsigned short* r2 = img0.row<const unsigned short>(2);
        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
                v4f32 _r00 = bfloat2float_msa(r0);
                v4f32 _r01 = bfloat2float_msa(r0 + 4);
                v4f32 _r02 = bfloat2float_msa(r0 + 8);
                v4f32 _r10 = bfloat2float_msa(r1);
                v4f32 _r11 = bfloat2float_msa(r1 + 4);
                v4f32 _r12 = bfloat2float_msa(r1 + 8);
                v4f32 _r20 = bfloat2float_msa(r2);
                v4f32 _r21 = bfloat2float_msa(r2 + 4);
                v4f32 _r22 = bfloat2float_msa(r2 + 8);

                v4f32 _max00 = __msa_fmax_w(_r00, _r01);
                _max00 = __msa_fmax_w(_max00, _r02);
                _max00 = __msa_fmax_w(_max00, _r10);
                _max00 = __msa_fmax_w(_max00, _r11);
                v4f32 _max01 = __msa_fmax_w(_r12, _r20);
                _max01 = __msa_fmax_w(_max01, _r21);
                _max01 = __msa_fmax_w(_max01, _r22);

                v4f32 _r03 = bfloat2float_msa(r0 + 12);
                v4f32 _r04 = bfloat2float_msa(r0 + 16);
                v4f32 _r13 = bfloat2float_msa(r1 + 12);
                v4f32 _r14 = bfloat2float_msa(r1 + 16);
                v4f32 _r23 = bfloat2float_msa(r2 + 12);
                v4f32 _r24 = bfloat2float_msa(r2 + 16);

                __msa_storel_d(float2bfloat_msa(__msa_fmax_w(_max00, _max01)), outptr);

                v4f32 _max10 = __msa_fmax_w(_r03, _r04);
                _max10 = __msa_fmax_w(_max10, _r02);
                _max10 = __msa_fmax_w(_max10, _r13);
                _max10 = __msa_fmax_w(_max10, _r14);
                v4f32 _max11 = __msa_fmax_w(_r12, _r23);
                _max10 = __msa_fmax_w(_max10, _r24);
                _max10 = __msa_fmax_w(_max10, _r22);

                __msa_storel_d(float2bfloat_msa(__msa_fmax_w(_max10, _max11)), outptr + 4);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr += 8;
            }

            for (; j < outw; j++)
            {
                v4f32 _r00 = bfloat2float_msa(r0);
                v4f32 _r01 = bfloat2float_msa(r0 + 4);
                v4f32 _r02 = bfloat2float_msa(r0 + 8);
                v4f32 _r10 = bfloat2float_msa(r1);
                v4f32 _r11 = bfloat2float_msa(r1 + 4);
                v4f32 _r12 = bfloat2float_msa(r1 + 8);
                v4f32 _r20 = bfloat2float_msa(r2);
                v4f32 _r21 = bfloat2float_msa(r2 + 4);
                v4f32 _r22 = bfloat2float_msa(r2 + 8);

                v4f32 _max0 = __msa_fmax_w(_r00, _r01);
                _max0 = __msa_fmax_w(_max0, _r02);
                _max0 = __msa_fmax_w(_max0, _r10);
                _max0 = __msa_fmax_w(_max0, _r11);
                v4f32 _max1 = __msa_fmax_w(_r12, _r20);
                _max1 = __msa_fmax_w(_max1, _r21);
                _max1 = __msa_fmax_w(_max1, _r22);

                __msa_storel_d(float2bfloat_msa(__msa_fmax_w(_max0, _max1)), outptr);

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
#endif // __mips_msa

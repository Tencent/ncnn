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
                v8i16 _zero_bf16 = __msa_fill_h(0);
                v8i16 _r001 = (v8i16)__msa_ld_h(r0, 0);
                v4f32 _r00 = (v4f32)__msa_ilvr_h(_r001, _zero_bf16);
                v4f32 _r01 = (v4f32)__msa_ilvl_h(_r001, _zero_bf16);
                v4f32 _r02 = bfloat2float_msa(r0 + 8);
                v8i16 _r101 = (v8i16)__msa_ld_h(r1, 0);
                v4f32 _r10 = (v4f32)__msa_ilvr_h(_r101, _zero_bf16);
                v4f32 _r11 = (v4f32)__msa_ilvl_h(_r101, _zero_bf16);
                v4f32 _r12 = bfloat2float_msa(r1 + 8);
                v8i16 _r201 = (v8i16)__msa_ld_h(r2, 0);
                v4f32 _r20 = (v4f32)__msa_ilvr_h(_r201, _zero_bf16);
                v4f32 _r21 = (v4f32)__msa_ilvl_h(_r201, _zero_bf16);
                v4f32 _r22 = bfloat2float_msa(r2 + 8);

                v4f32 _max00 = __msa_fmax_w(_r00, _r01);
                _max00 = __msa_fmax_w(_max00, _r02);
                _max00 = __msa_fmax_w(_max00, _r10);
                _max00 = __msa_fmax_w(_max00, _r11);
                v4f32 _max01 = __msa_fmax_w(_r12, _r20);
                _max01 = __msa_fmax_w(_max01, _r21);
                _max01 = __msa_fmax_w(_max01, _r22);

                v8i16 _r034_bf16 = __msa_ld_h(r0 + 12, 0);
                v4f32 _r03 = (v4f32)__msa_ilvr_h(_r034_bf16, _zero_bf16);
                v4f32 _r04 = (v4f32)__msa_ilvl_h(_r034_bf16, _zero_bf16);
                v8i16 _r134_bf16 = __msa_ld_h(r1 + 12, 0);
                v4f32 _r13 = (v4f32)__msa_ilvr_h(_r134_bf16, _zero_bf16);
                v4f32 _r14 = (v4f32)__msa_ilvl_h(_r134_bf16, _zero_bf16);
                v8i16 _r234_bf16 = __msa_ld_h(r2 + 12, 0);
                v4f32 _r23 = (v4f32)__msa_ilvr_h(_r234_bf16, _zero_bf16);
                v4f32 _r24 = (v4f32)__msa_ilvl_h(_r234_bf16, _zero_bf16);

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
                v8i16 _zero_bf16 = __msa_fill_h(0);
                v8i16 _r001 = (v8i16)__msa_ld_h(r0, 0);
                v4f32 _r00 = (v4f32)__msa_ilvr_h(_r001, _zero_bf16);
                v4f32 _r01 = (v4f32)__msa_ilvl_h(_r001, _zero_bf16);
                v4f32 _r02 = bfloat2float_msa(r0 + 8);
                v8i16 _r101 = (v8i16)__msa_ld_h(r1, 0);
                v4f32 _r10 = (v4f32)__msa_ilvr_h(_r101, _zero_bf16);
                v4f32 _r11 = (v4f32)__msa_ilvl_h(_r101, _zero_bf16);
                v4f32 _r12 = bfloat2float_msa(r1 + 8);
                v8i16 _r201 = (v8i16)__msa_ld_h(r2, 0);
                v4f32 _r20 = (v4f32)__msa_ilvr_h(_r201, _zero_bf16);
                v4f32 _r21 = (v4f32)__msa_ilvl_h(_r201, _zero_bf16);
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

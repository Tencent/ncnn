// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __mips_msa
static void pooling2x2s2_max_pack4_bf16s_msa(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
                v8i16 _zero_bf16 = __msa_fill_h(0);
                v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                v8i16 _r1 = (v8i16)__msa_ld_h(r1, 0);
                v4f32 _r00 = (v4f32)__msa_ilvr_h(_r0, _zero_bf16);
                v4f32 _r01 = (v4f32)__msa_ilvl_h(_r0, _zero_bf16);
                v4f32 _r10 = (v4f32)__msa_ilvr_h(_r1, _zero_bf16);
                v4f32 _r11 = (v4f32)__msa_ilvl_h(_r1, _zero_bf16);

                v4f32 _max0 = __msa_fmax_w(_r00, _r01);
                v4f32 _max1 = __msa_fmax_w(_r10, _r11);
                v4f32 _max = __msa_fmax_w(_max0, _max1);

                *(int64_t*)outptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_max), 0);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
#endif // __mips_msa

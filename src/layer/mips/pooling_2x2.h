// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pooling2x2s2_max_msa(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r11 = (v4f32)__msa_ld_w(r1 + 4, 0);

                v4f32 _max0 = __msa_fmax_w(_r00, _r10);
                v4f32 _max1 = __msa_fmax_w(_r01, _r11);
                _max0 = __msa_fmax_w(_max0, (v4f32)__msa_shf_w((v4i32)_max0, _MSA_SHUFFLE(2, 3, 0, 1)));
                _max1 = __msa_fmax_w(_max1, (v4f32)__msa_shf_w((v4i32)_max1, _MSA_SHUFFLE(2, 3, 0, 1)));
                v4f32 _max = (v4f32)__msa_pckev_w((v4i32)_max1, (v4i32)_max0);

                __msa_st_w((v4i32)_max, outptr, 0);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            for (; j < outw; j++)
            {
                float max0 = std::max(r0[0], r0[1]);
                float max1 = std::max(r1[0], r1[1]);

                *outptr = std::max(max0, max1);

                r0 += 2;
                r1 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}

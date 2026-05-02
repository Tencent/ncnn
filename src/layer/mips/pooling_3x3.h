// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pooling3x3s2_max_msa(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
                v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r04 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r10 = (v4f32)__msa_ld_w(r1, 0);
                v4f32 _r14 = (v4f32)__msa_ld_w(r1 + 4, 0);
                v4f32 _r20 = (v4f32)__msa_ld_w(r2, 0);
                v4f32 _r24 = (v4f32)__msa_ld_w(r2 + 4, 0);

                v4f32 _r0max0 = __msa_fmax_w(_r00, (v4f32)__msa_shf_w((v4i32)_r00, _MSA_SHUFFLE(2, 3, 0, 1)));
                v4f32 _r0max1 = __msa_fmax_w(_r04, (v4f32)__msa_shf_w((v4i32)_r04, _MSA_SHUFFLE(2, 3, 0, 1)));
                v4f32 _r0max = (v4f32)__msa_pckev_w((v4i32)_r0max1, (v4i32)_r0max0);
                v4i32 _r0v2 = __msa_shf_w(__msa_pckev_w((v4i32)_r04, (v4i32)_r00), _MSA_SHUFFLE(0, 3, 2, 1));
                _r0v2 = __msa_insert_w(_r0v2, 3, ((const int*)r0)[8]);
                _r0max = __msa_fmax_w(_r0max, (v4f32)_r0v2);

                v4f32 _r1max0 = __msa_fmax_w(_r10, (v4f32)__msa_shf_w((v4i32)_r10, _MSA_SHUFFLE(2, 3, 0, 1)));
                v4f32 _r1max1 = __msa_fmax_w(_r14, (v4f32)__msa_shf_w((v4i32)_r14, _MSA_SHUFFLE(2, 3, 0, 1)));
                v4f32 _r1max = (v4f32)__msa_pckev_w((v4i32)_r1max1, (v4i32)_r1max0);
                v4i32 _r1v2 = __msa_shf_w(__msa_pckev_w((v4i32)_r14, (v4i32)_r10), _MSA_SHUFFLE(0, 3, 2, 1));
                _r1v2 = __msa_insert_w(_r1v2, 3, ((const int*)r1)[8]);
                _r1max = __msa_fmax_w(_r1max, (v4f32)_r1v2);

                v4f32 _r2max0 = __msa_fmax_w(_r20, (v4f32)__msa_shf_w((v4i32)_r20, _MSA_SHUFFLE(2, 3, 0, 1)));
                v4f32 _r2max1 = __msa_fmax_w(_r24, (v4f32)__msa_shf_w((v4i32)_r24, _MSA_SHUFFLE(2, 3, 0, 1)));
                v4f32 _r2max = (v4f32)__msa_pckev_w((v4i32)_r2max1, (v4i32)_r2max0);
                v4i32 _r2v2 = __msa_shf_w(__msa_pckev_w((v4i32)_r24, (v4i32)_r20), _MSA_SHUFFLE(0, 3, 2, 1));
                _r2v2 = __msa_insert_w(_r2v2, 3, ((const int*)r2)[8]);
                _r2max = __msa_fmax_w(_r2max, (v4f32)_r2v2);

                v4f32 _max = __msa_fmax_w(_r0max, _r1max);
                _max = __msa_fmax_w(_max, _r2max);
                __msa_st_w((v4i32)_max, outptr, 0);

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

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bnll_mips.h"

#if __mips_msa
#include <msa.h>
#include "mips_usability.h"
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

BNLL_mips::BNLL_mips()
{
#if __mips_msa
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int BNLL_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __mips_msa
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);

            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _abs_p = (v4f32)__msa_bclri_w((v4u32)_p, 31);
            v4f32 _tmp = log_ps(__msa_fadd_w(_one, exp_ps((v4f32)__msa_bnegi_w((v4u32)_abs_p, 31))));
            v4f32 _outp = __msa_fadd_w(__msa_fmax_w(_p, _zero), _tmp);
            __msa_st_w((v4i32)_outp, ptr, 0);

            ptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            if (*ptr > 0.f)
                *ptr = *ptr + logf(1.f + expf(-*ptr));
            else
                *ptr = logf(1.f + expf(*ptr));

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int BNLL_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __mips_msa
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _abs_p = (v4f32)__msa_bclri_w((v4u32)_p, 31);
            v4f32 _tmp = log_ps(__msa_fadd_w(_one, exp_ps((v4f32)__msa_bnegi_w((v4u32)_abs_p, 31))));
            v4f32 _outp = __msa_fadd_w(__msa_fmax_w(_p, _zero), _tmp);
            float2bfloat_msa_store(_outp, ptr);

            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v > 0.f)
                v = v + logf(1.f + expf(-v));
            else
                v = logf(1.f + expf(v));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

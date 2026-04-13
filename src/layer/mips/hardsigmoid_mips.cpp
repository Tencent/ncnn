// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardsigmoid_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

HardSigmoid_mips::HardSigmoid_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int HardSigmoid_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        v4f32 _alpha = (v4f32)__msa_fill_w_f32(alpha);
        v4f32 _beta = (v4f32)__msa_fill_w_f32(beta);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = __msa_fmadd_w(_beta, _p, _alpha);
            _p = __msa_fmax_w(_p, _zero);
            _p = __msa_fmin_w(_p, _one);
            __msa_st_w((v4i32)_p, ptr, 0);

            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                *ptr = 1.f;
            else
                *ptr = *ptr * alpha + beta;
            ++ptr;
        }
    }

    return 0;
}

#if NCNN_BF16
int HardSigmoid_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        v4f32 _alpha = (v4f32)__msa_fill_w_f32(alpha);
        v4f32 _beta = (v4f32)__msa_fill_w_f32(beta);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            _p = __msa_fmadd_w(_beta, _p, _alpha);
            _p = __msa_fmax_w(_p, _zero);
            _p = __msa_fmin_w(_p, _one);
            float2bfloat_msa_store(_p, ptr);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < lower)
                v = 0.f;
            else if (v > upper)
                v = 1.f;
            else
                v = v * alpha + beta;
            *ptr = float32_to_bfloat16(v);
            ++ptr;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

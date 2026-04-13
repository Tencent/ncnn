// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_mips.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

#if NCNN_BF16
#include "selu_bf16s.h"
#endif

SELU_mips::SELU_mips()
{
#if __mips_msa
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
#endif // __mips_msa
}

int SELU_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;
    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __mips_msa
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        v4f32 _alphaxlambda = (v4f32)__msa_fill_w_f32(alphaxlambda);
        v4f32 _lambda = (v4f32)__msa_fill_w_f32(lambda);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4i32_w _lemask = __msa_fcle_w(_p, _zero);

            v4f32 _nps = exp_ps(_p);
            _nps = __msa_fsub_w(_nps, _one);
            _nps = __msa_fmul_w(_nps, _alphaxlambda);

            _p = __msa_fmul_w(_p, _lambda);

            _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_nps);
            __msa_st_w((v4i32)_p, ptr, 0);

            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = (expf(*ptr) - 1.f) * alphaxlambda;
            else
                *ptr *= lambda;

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int SELU_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    selu_bf16s(bottom_top_blob, alpha, lambda, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

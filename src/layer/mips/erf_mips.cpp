// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "erf_mips.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

#if NCNN_BF16
#include "erf_bf16s.h"
#endif

Erf_mips::Erf_mips()
{
#if __mips_msa
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
#endif
}

int Erf_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = erf_ps(_p);
            __msa_st_w((v4i32)_p, ptr, 0);

            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = erff(*ptr);
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Erf_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    erf_bf16s(bottom_top_blob, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

// Copyright 2019 Leo <leo@nullptr.com.cn>
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bias_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

namespace ncnn {

int Bias_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = bias_ptr[q];

        int i = 0;
#if __mips_msa
        v4f32 _bias = (v4f32)__msa_fill_w_f32(bias);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _outp = __msa_fadd_w(_p, _bias);
            __msa_st_w((v4i32)_outp, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = *ptr + bias;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn

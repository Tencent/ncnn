// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dropout_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Dropout_mips::Dropout_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

int Dropout_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

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
        v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = __msa_fmul_w(_p, _scale);
            __msa_st_w((v4i32)_p, ptr, 0);

            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = *ptr * scale;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn

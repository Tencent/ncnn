// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "erf_loongarch.h"

namespace ncnn {

Erf_loongarch::Erf_loongarch()
{
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Erf_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = opt.workspace_allocator;

        Mat bottom_top_blob_fp32;
        cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
        if (bottom_top_blob_fp32.empty())
            return -100;

        Option opt_fp32 = opt;
        opt_fp32.use_bf16_storage = false;
        opt_fp32.use_bf16_packed = false;

        int ret = forward_inplace(bottom_top_blob_fp32, opt_fp32);
        if (ret != 0)
            return ret;

        cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
        if (bottom_top_blob.empty())
            return -100;

        return 0;
    }
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
        for (int i = 0; i < size; i++)
        {
            *ptr = erff(*ptr);
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn

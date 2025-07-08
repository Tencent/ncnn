// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "softplus.h"

namespace ncnn {

Softplus::Softplus()
{
    one_blob_only = true;
    support_inplace = true;
}

int Softplus::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        for (int i = 0; i < size; i++)
        {
            ptr[i] = logf(expf(ptr[i]) + 1.0f);
        }
    }

    return 0;
}

} // namespace ncnn

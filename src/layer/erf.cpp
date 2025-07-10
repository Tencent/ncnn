// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "erf.h"

namespace ncnn {

Erf::Erf()
{
    one_blob_only = true;
    support_inplace = true;
}

int Erf::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
            ptr[i] = erff(ptr[i]);
        }
    }

    return 0;
}

} // namespace ncnn

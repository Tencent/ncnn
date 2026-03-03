// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "threshold.h"

namespace ncnn {

Threshold::Threshold()
{
    one_blob_only = true;
    support_inplace = true;
}

int Threshold::load_param(const ParamDict& pd)
{
    threshold = pd.get(0, 0.f);

    return 0;
}

int Threshold::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
            ptr[i] = ptr[i] > threshold ? 1.f : 0.f;
        }
    }

    return 0;
}

} // namespace ncnn

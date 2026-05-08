// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "power.h"

namespace ncnn {

Power::Power()
{
    one_blob_only = true;
    support_inplace = true;
}

int Power::load_param(const ParamDict& pd)
{
    power = pd.get(0, 1.f);
    scale = pd.get(1, 1.f);
    shift = pd.get(2, 0.f);

    return 0;
}

int Power::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
            ptr[i] = powf((shift + ptr[i] * scale), power);
        }
    }

    return 0;
}

} // namespace ncnn

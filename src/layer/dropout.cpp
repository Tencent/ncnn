// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dropout.h"

namespace ncnn {

Dropout::Dropout()
{
    one_blob_only = true;
    support_inplace = true;
}

int Dropout::load_param(const ParamDict& pd)
{
    scale = pd.get(0, 1.f);

    return 0;
}

int Dropout::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

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
            ptr[i] = ptr[i] * scale;
        }
    }

    return 0;
}

} // namespace ncnn

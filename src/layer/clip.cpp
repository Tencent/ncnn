// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip.h"

#include <float.h>

namespace ncnn {

Clip::Clip()
{
    one_blob_only = true;
    support_inplace = true;
}

int Clip::load_param(const ParamDict& pd)
{
    min = pd.get(0, -FLT_MAX);
    max = pd.get(1, FLT_MAX);

    return 0;
}

int Clip::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < min)
                ptr[i] = min;
            if (ptr[i] > max)
                ptr[i] = max;
        }
    }

    return 0;
}

} // namespace ncnn

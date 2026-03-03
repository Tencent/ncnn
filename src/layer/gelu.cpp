// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu.h"

namespace ncnn {

GELU::GELU()
{
    one_blob_only = true;
    support_inplace = true;
}

int GELU::load_param(const ParamDict& pd)
{
    fast_gelu = pd.get(0, 0);

    return 0;
}

int GELU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    if (fast_gelu)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
                ptr[i] = 0.5f * ptr[i] * (1.0f + tanhf(0.79788452f * (ptr[i] + 0.044715f * ptr[i] * ptr[i] * ptr[i])));
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                // y = x * P(X <= x) where X ~ N(0, 1)
                ptr[i] = 0.5f * ptr[i] * erfcf(-0.70710678f * ptr[i]);
            }
        }
    }

    return 0;
}

} // namespace ncnn

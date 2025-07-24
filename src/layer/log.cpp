// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "log.h"

namespace ncnn {

Log::Log()
{
    one_blob_only = true;
    support_inplace = true;
}

int Log::load_param(const ParamDict& pd)
{
    base = pd.get(0, -1.f);
    scale = pd.get(1, 1.f);
    shift = pd.get(2, 0.f);

    return 0;
}

int Log::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (base == -1.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] = logf(shift + ptr[i] * scale);
            }
        }
    }
    else
    {
        float log_base_inv = 1.f / logf(base);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] = logf(shift + ptr[i] * scale) * log_base_inv;
            }
        }
    }

    return 0;
}

} // namespace ncnn

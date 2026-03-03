// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardswish.h"

namespace ncnn {

HardSwish::HardSwish()
{
    one_blob_only = true;
    support_inplace = true;
}

int HardSwish::load_param(const ParamDict& pd)
{
    // Note that tensorflow/pytorch use alpha,beta = 1/6, 0.5, not the default value here.
    // You can setup them manually in .param file.
    alpha = pd.get(0, 0.2f);
    beta = pd.get(1, 0.5f);
    lower = -beta / alpha;
    upper = (1.f / alpha) + lower;

    return 0;
}

int HardSwish::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
            if (ptr[i] < lower)
                ptr[i] = 0.f;
            else if (ptr[i] > upper)
                ;
            else
                ptr[i] = ptr[i] * (ptr[i] * alpha + beta);
        }
    }

    return 0;
}

} // namespace ncnn

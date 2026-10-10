// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu.h"

namespace ncnn {

SELU::SELU()
{
    one_blob_only = true;
    support_inplace = true;
}

int SELU::load_param(const ParamDict& pd)
{
    alpha = pd.get(0, 1.67326324f);
    lambda = pd.get(1, 1.050700987f);

    return 0;
}

int SELU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;
    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = (expf(ptr[i]) - 1.f) * alphaxlambda;
            else
                ptr[i] *= lambda;
        }
    }

    return 0;
}

} // namespace ncnn

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bias.h"

namespace ncnn {

Bias::Bias()
{
    one_blob_only = true;
    support_inplace = true;
}

int Bias::load_param(const ParamDict& pd)
{
    bias_data_size = pd.get(0, 0);

    return 0;
}

int Bias::load_model(const ModelBin& mb)
{
    bias_data = mb.load(bias_data_size, 1);
    if (bias_data.empty())
        return -100;

    return 0;
}

int Bias::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

        float bias = bias_data[q];

        for (int i = 0; i < size; i++)
        {
            ptr[i] += bias;
        }
    }

    return 0;
}

} // namespace ncnn

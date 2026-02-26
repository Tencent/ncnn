// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm.h"

namespace ncnn {

InstanceNorm::InstanceNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int InstanceNorm::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.001f);
    affine = pd.get(2, 1);

    return 0;
}

int InstanceNorm::load_model(const ModelBin& mb)
{
    if (affine == 0)
        return 0;

    gamma_data = mb.load(channels, 1);
    if (gamma_data.empty())
        return -100;

    beta_data = mb.load(channels, 1);
    if (beta_data.empty())
        return -100;

    return 0;
}

int InstanceNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;
        for (int i = 0; i < size; i++)
        {
            sum += ptr[i];
            //sqsum += ptr[i] * ptr[i];
        }
        float mean = sum / size;
        float tmp = 0.f;
        for (int i = 0; i < size; i++)
        {
            tmp = ptr[i] - mean;
            sqsum += tmp * tmp;
        }
        float var = sqsum / size;
        // the var maybe minus due to accuracy
        //float var = sqsum / size - mean * mean;

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / (sqrtf(var + eps));
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / (sqrtf(var + eps));
            b = -mean * a;
        }

        for (int i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] * a + b;
        }
    }

    return 0;
}

} // namespace ncnn

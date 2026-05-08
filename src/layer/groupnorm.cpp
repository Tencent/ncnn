// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm.h"

namespace ncnn {

GroupNorm::GroupNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int GroupNorm::load_param(const ParamDict& pd)
{
    group = pd.get(0, 1);
    channels = pd.get(1, 0);
    eps = pd.get(2, 0.001f);
    affine = pd.get(3, 1);

    return 0;
}

int GroupNorm::load_model(const ModelBin& mb)
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

static void groupnorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, size_t cstep)
{
    float sum = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q;
        for (int i = 0; i < size; i++)
        {
            sum += ptr0[i];
        }
    }

    float mean = sum / (channels * size);

    float sqsum = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q;
        for (int i = 0; i < size; i++)
        {
            float v = ptr0[i] - mean;
            sqsum += v * v;
        }
    }

    float var = sqsum / (channels * size);

    float a = 1.f / sqrtf(var + eps);
    float b = -mean * a;

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q;
            const float gamma = gamma_ptr[q];
            const float beta = beta_ptr[q];
            for (int i = 0; i < size; i++)
            {
                ptr0[i] = (ptr0[i] * a + b) * gamma + beta;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q;
            for (int i = 0; i < size; i++)
            {
                ptr0[i] = ptr0[i] * a + b;
            }
        }
    }
}

int GroupNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int channels_g = channels / group;

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g, 1, 1);
        }
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.row_range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g, w, w);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d;
        const size_t cstep = bottom_top_blob.cstep;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.channel_range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g, size, cstep);
        }
    }

    return 0;
}

} // namespace ncnn

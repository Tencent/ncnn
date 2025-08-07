// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm.h"

namespace ncnn {

LayerNorm::LayerNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int LayerNorm::load_param(const ParamDict& pd)
{
    affine_size = pd.get(0, 0);
    eps = pd.get(1, 0.001f);
    affine = pd.get(2, 1);

    return 0;
}

int LayerNorm::load_model(const ModelBin& mb)
{
    if (affine == 0)
        return 0;

    gamma_data = mb.load(affine_size, 1);
    if (gamma_data.empty())
        return -100;

    beta_data = mb.load(affine_size, 1);
    if (beta_data.empty())
        return -100;

    return 0;
}

static void layernorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int size)
{
    float sum = 0.f;
    for (int i = 0; i < size; i++)
    {
        sum += ptr[i];
    }

    float mean = sum / size;

    float sqsum = 0.f;
    for (int i = 0; i < size; i++)
    {
        float v = ptr[i] - mean;
        sqsum += v * v;
    }

    float var = sqsum / size;

    float a = 1.f / sqrtf(var + eps);
    float b = -mean * a;

    if (gamma_ptr && beta_ptr)
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = (ptr[i] * a + b) * gamma_ptr[i] + beta_ptr[i];
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] * a + b;
        }
    }
}

int LayerNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / sqrt(var + eps) * gamma + beta

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        // assert affine_size == w

        float* ptr = bottom_top_blob;
        layernorm(ptr, gamma_data, beta_data, eps, w);
    }
    else if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm(ptr, gamma_data, beta_data, eps, w);
        }
    }
    else if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;

        int group_size;
        int num_groups_per_channel;

        if (affine_size == w)
        {
            group_size = w;
            num_groups_per_channel = h;
        }
        else // if (affine_size == w * h)
        {
            group_size = w * h;
            num_groups_per_channel = 1;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* channel_ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < num_groups_per_channel; i++)
            {
                float* ptr = channel_ptr + i * group_size;
                layernorm(ptr, gamma_data, beta_data, eps, group_size);
            }
        }
    }

    return 0;
}

} // namespace ncnn
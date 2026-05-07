// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rmsnorm.h"

namespace ncnn {

RMSNorm::RMSNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int RMSNorm::load_param(const ParamDict& pd)
{
    affine_size = pd.get(0, 0);
    eps = pd.get(1, 0.001f);
    affine = pd.get(2, 1);

    return 0;
}

int RMSNorm::load_model(const ModelBin& mb)
{
    if (affine == 0)
        return 0;

    gamma_data = mb.load(affine_size, 1);
    if (gamma_data.empty())
        return -100;

    return 0;
}

static void rmsnorm(float* ptr, const float* gamma_ptr, float eps, int size)
{
    float sqsum = 0.f;
    for (int i = 0; i < size; i++)
    {
        sqsum += ptr[i] * ptr[i];
    }

    float rms = sqsum / size;

    float a = 1.f / sqrtf(rms + eps);

    if (gamma_ptr)
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = (ptr[i] * a) * gamma_ptr[i];
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] * a;
        }
    }
}

int RMSNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // x = x / sqrt(rms + eps) * gamma

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        // assert affine_size == w

        float* ptr = bottom_top_blob;
        rmsnorm(ptr, gamma_data, eps, w);
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm(ptr, gamma_data, eps, w);
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    rmsnorm(ptr, gamma_data, eps, w);
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm(ptr, gamma_data, eps, w * h);
            }
        }
    }

    return 0;
}

} // namespace ncnn

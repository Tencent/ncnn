// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mvn.h"

namespace ncnn {

MVN::MVN()
{
    one_blob_only = true;
    support_inplace = false;
}

int MVN::load_param(const ParamDict& pd)
{
    normalize_variance = pd.get(0, 0);
    across_channels = pd.get(1, 0);
    eps = pd.get(2, 0.0001f);

    return 0;
}

int MVN::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // prepare sum per channel
    Mat sum(channels, elemsize, opt.workspace_allocator);
    if (sum.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);

        float s = 0.f;
        for (int i = 0; i < size; i++)
        {
            s += ptr[i];
        }

        sum[q] = s;
    }

    if (across_channels)
    {
        // compute mean across channels
        float mean = 0.f;
        for (int q = 0; q < channels; q++)
        {
            mean += sum[q];
        }
        mean = mean / (channels * size);

        // subtract mean
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] - mean;
            }
        }
    }
    else
    {
        // subtract mean
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            float mean = sum[q] / size;

            for (int i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] - mean;
            }
        }
    }

    if (normalize_variance)
    {
        // prepare squared sum per channel
        Mat sqsum(channels, elemsize, opt.workspace_allocator);
        if (sqsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = top_blob.channel(q);

            float s = 0.f;
            for (int i = 0; i < size; i++)
            {
                s += ptr[i] * ptr[i];
            }

            sqsum[q] = s;
        }

        if (across_channels)
        {
            // compute squared mean across channels
            float sqmean = 0.f;
            for (int q = 0; q < channels; q++)
            {
                sqmean += sqsum[q];
            }
            sqmean = sqmean / (channels * size);

            // normalize variance
            float norm_var = sqrtf(sqmean) + eps;
            float norm_var_inv = 1.f / norm_var;

            // apply normalize_variance
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = outptr[i] * norm_var_inv;
                }
            }
        }
        else
        {
            // apply normalize_variance
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = top_blob.channel(q);
                float sqmean = sqsum[q] / size;
                float norm_var = sqrt(sqmean) + eps;
                float norm_var_inv = 1.f / norm_var;

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = outptr[i] * norm_var_inv;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu.h"

namespace ncnn {

PReLU::PReLU()
{
    one_blob_only = true;
    support_inplace = true;
}

int PReLU::load_param(const ParamDict& pd)
{
    num_slope = pd.get(0, 0);

    return 0;
}

int PReLU::load_model(const ModelBin& mb)
{
    slope_data = mb.load(num_slope, 1);
    if (slope_data.empty())
        return -100;

    return 0;
}

int PReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope_data[i];
            }
        }
        else
        {
            float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            for (int j = 0; j < w; j++)
            {
                if (ptr[j] < 0)
                    ptr[j] *= slope;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

    return 0;
}

} // namespace ncnn

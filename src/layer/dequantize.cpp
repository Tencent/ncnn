// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dequantize.h"

namespace ncnn {

Dequantize::Dequantize()
{
    one_blob_only = true;
    support_inplace = false;
}

int Dequantize::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 1);
    bias_data_size = pd.get(1, 0);

    return 0;
}

int Dequantize::load_model(const ModelBin& mb)
{
    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    if (bias_data_size)
    {
        bias_data = mb.load(bias_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

static void dequantize(const int* intptr, float* ptr, float scale, float bias, int size)
{
    for (int i = 0; i < size; i++)
    {
        *ptr = *intptr * scale + bias;
        intptr++;
        ptr++;
    }
}

int Dequantize::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 1)
    {
        // assert scale_data_size == 1
        // assert bias_data_size == 0 || bias_data_size == 1

        const int* intptr = bottom_blob;
        float* ptr = top_blob;

        const float scale = scale_data[0];
        const float bias = bias_data_size == 0 ? 0.f : bias_data[0];

        dequantize(intptr, ptr, scale, bias, w);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            float* ptr = top_blob.row(i);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
            const float bias = bias_data_size == 0 ? 0.f : bias_data_size == 1 ? bias_data[0] : bias_data[i];

            dequantize(intptr, ptr, scale, bias, w);
        }
    }

    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            float* ptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
            const float bias = bias_data_size == 0 ? 0.f : bias_data_size == 1 ? bias_data[0] : bias_data[q];

            dequantize(intptr, ptr, scale, bias, w * h);
        }
    }

    return 0;
}

} // namespace ncnn

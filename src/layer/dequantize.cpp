// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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

int Dequantize::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        float* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale;
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias_data[i];
                }
            }
        }
        else
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i];
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias_data[i];
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                for (int j = 0; j < w; j++)
                {
                    ptr[j] = intptr[j] * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                for (int j = 0; j < w; j++)
                {
                    ptr[j] = intptr[j] * scale + bias;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = intptr[i] * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn

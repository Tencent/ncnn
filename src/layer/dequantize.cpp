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
    support_inplace = true;
}

int Dequantize::load_param(const ParamDict& pd)
{
    scale = pd.get(0, 1.f);
    bias_term = pd.get(1, 0);
    bias_data_size = pd.get(2, 0);

    return 0;
}

int Dequantize::load_model(const ModelBin& mb)
{
    if (bias_term)
    {
        bias_data = mb.load(bias_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Dequantize::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        const int* intptr = bottom_top_blob;
        float* ptr = bottom_top_blob;

        if (bias_term)
        {
            if (bias_data_size > 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias_data[i];
                }
            }
            else
            {
                float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                ptr[i] = intptr[i] * scale;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_top_blob.row<const int>(i);
                float* ptr = bottom_top_blob.row(i);

                float bias = bias_data_size > 1 ? bias_data[i] : bias_data[0];

                for (int j = 0; j < w; j++)
                {
                    ptr[j] = intptr[j] * scale + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_top_blob.row<const int>(i);
                float* ptr = bottom_top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    ptr[j] = intptr[j] * scale;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_top_blob.channel(q);
                float* ptr = bottom_top_blob.channel(q);

                float bias = bias_data_size > 1 ? bias_data[q] : bias_data[0];

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_top_blob.channel(q);
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = intptr[i] * scale;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn

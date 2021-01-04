// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "normalize.h"

#include <math.h>

namespace ncnn {

Normalize::Normalize()
{
    one_blob_only = true;
    support_inplace = true;
}

int Normalize::load_param(const ParamDict& pd)
{
    across_spatial = pd.get(0, 0);
    across_channel = pd.get(4, 1);
    channel_shared = pd.get(1, 0);
    eps = pd.get(2, 0.0001f);
    eps_mode = pd.get(9, 0);
    scale_data_size = pd.get(3, 0);

    return 0;
}

int Normalize::load_model(const ModelBin& mb)
{
    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    return 0;
}

int Normalize::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int size = w * h;

    if (across_spatial && across_channel)
    {
        // square
        Mat square_sum_blob;
        square_sum_blob.create(channels, elemsize, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);

            float ssum = 0.f;
            for (int i = 0; i < size; i++)
            {
                ssum += ptr[i] * ptr[i];
            }

            square_sum_blob[q] = ssum;
        }

        float ssum = 0.f;
        for (int q = 0; q < channels; q++)
        {
            ssum += square_sum_blob[q];
        }

        float a;
        if (eps_mode == 0) // caffe/mxnet
        {
            a = static_cast<float>(1.f / sqrt(ssum + eps));
        }
        else if (eps_mode == 1) // pytorch
        {
            a = 1.f / std::max((float)sqrt(ssum), eps);
        }
        else //if (eps_mode == 2) // tensorflow
        {
            a = static_cast<float>(1.f / sqrt(std::max(ssum, eps)));
        }

        if (channel_shared)
        {
            float scale = a * scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float scale = a * scale_data[q];

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * scale;
                }
            }
        }

        return 0;
    }

    if (across_spatial && !across_channel)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float ssum = 0.f;
            for (int i = 0; i < size; i++)
            {
                ssum += ptr[i] * ptr[i];
            }

            float a;
            if (eps_mode == 0) // caffe/mxnet
            {
                a = static_cast<float>(1.f / sqrt(ssum + eps));
            }
            else if (eps_mode == 1) // pytorch
            {
                a = 1.f / std::max((float)sqrt(ssum), eps);
            }
            else //if (eps_mode == 2) // tensorflow
            {
                a = static_cast<float>(1.f / sqrt(std::max(ssum, eps)));
            }

            float scale = a * (channel_shared ? scale_data[0] : scale_data[q]);

            for (int i = 0; i < size; i++)
            {
                ptr[i] = ptr[i] * scale;
            }
        }

        return 0;
    }

    if (!across_spatial && across_channel)
    {
        // square sum, 1 / sqrt(ssum)
        Mat square_sum_blob;
        square_sum_blob.create(size, elemsize, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        if (channel_shared)
        {
            float scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                float ssum = 0.f;
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_top_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }

                float a;
                if (eps_mode == 0) // caffe/mxnet
                {
                    a = static_cast<float>(1.f / sqrt(ssum + eps));
                }
                else if (eps_mode == 1) // pytorch
                {
                    a = 1.f / std::max((float)sqrt(ssum), eps);
                }
                else //if (eps_mode == 2) // tensorflow
                {
                    a = static_cast<float>(1.f / sqrt(std::max(ssum, eps)));
                }

                square_sum_blob[i] = a * scale;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * square_sum_blob[i];
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                float ssum = 0.f;
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_top_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }

                float a;
                if (eps_mode == 0) // caffe/mxnet
                {
                    a = static_cast<float>(1.f / sqrt(ssum + eps));
                }
                else if (eps_mode == 1) // pytorch
                {
                    a = 1.f / std::max((float)sqrt(ssum), eps);
                }
                else //if (eps_mode == 2) // tensorflow
                {
                    a = static_cast<float>(1.f / sqrt(std::max(ssum, eps)));
                }

                square_sum_blob[i] = a;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float scale = scale_data[q];

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * square_sum_blob[i] * scale;
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn

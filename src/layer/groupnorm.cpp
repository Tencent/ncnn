// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "groupnorm.h"

#include <math.h>

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

int GroupNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / sqrt(var + eps) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;

    int channels_per_group = channels / group;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat bottom_top_blob_g = bottom_top_blob.channel_range(g * channels_per_group, channels_per_group);

        // mean and var
        float sum = 0.f;
        for (int q = 0; q < channels_per_group; q++)
        {
            const float* ptr = bottom_top_blob_g.channel(q);
            for (int i = 0; i < size; i++)
            {
                sum += ptr[i];
            }
        }
        float mean = sum / (channels_per_group * size);

        float sqsum = 0.f;
        for (int q = 0; q < channels_per_group; q++)
        {
            const float* ptr = bottom_top_blob_g.channel(q);
            for (int i = 0; i < size; i++)
            {
                float tmp = ptr[i] - mean;
                sqsum += tmp * tmp;
            }
        }
        float var = sqsum / (channels_per_group * size);

        for (int q = 0; q < channels_per_group; q++)
        {
            float a;
            float b;
            if (affine)
            {
                float gamma = gamma_data[g * channels_per_group + q];
                float beta = beta_data[g * channels_per_group + q];

                a = static_cast<float>(gamma / sqrt(var + eps));
                b = -mean * a + beta;
            }
            else
            {
                a = static_cast<float>(1.f / (sqrt(var + eps)));
                b = -mean * a;
            }

            float* ptr = bottom_top_blob_g.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] = ptr[i] * a + b;
            }
        }
    }

    return 0;
}

} // namespace ncnn

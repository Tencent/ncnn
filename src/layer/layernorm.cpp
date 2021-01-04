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

#include "layernorm.h"

#include <math.h>

namespace ncnn {

LayerNorm::LayerNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int LayerNorm::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.001f);

    return 0;
}

int LayerNorm::load_model(const ModelBin& mb)
{
    gamma_data = mb.load(channels, 1);
    if (gamma_data.empty())
        return -100;

    beta_data = mb.load(channels, 1);
    if (beta_data.empty())
        return -100;

    return 0;
}

int LayerNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / sqrt(var + eps) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;

    // mean and var
    float sum = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        for (int i = 0; i < size; i++)
        {
            sum += ptr[i];
        }
    }
    float mean = sum / (channels * size);

    float sqsum = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        for (int i = 0; i < size; i++)
        {
            float tmp = ptr[i] - mean;
            sqsum += tmp * tmp;
        }
    }
    float var = sqsum / (channels * size);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float gamma = gamma_data[q];
        float beta = beta_data[q];

        float a = static_cast<float>(gamma / sqrt(var + eps));
        float b = -mean * a + beta;

        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] * a + b;
        }
    }

    return 0;
}

} // namespace ncnn

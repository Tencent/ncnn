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

#include "batchnorm.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm)

BatchNorm::BatchNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int BatchNorm::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);

    return 0;
}

int BatchNorm::load_model(const ModelBin& mb)
{
    slope_data = mb.load(channels, 1);
    if (slope_data.empty())
        return -100;

    mean_data = mb.load(channels, 1);
    if (mean_data.empty())
        return -100;

    var_data = mb.load(channels, 1);
    if (var_data.empty())
        return -100;

    bias_data = mb.load(channels, 1);
    if (bias_data.empty())
        return -100;

    a_data.create(channels);
    if (a_data.empty())
        return -100;
    b_data.create(channels);
    if (b_data.empty())
        return -100;

    for (int i=0; i<channels; i++)
    {
        float sqrt_var = sqrt(var_data[i]);
        a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
        b_data[i] = slope_data[i] / sqrt_var;
    }

    return 0;
}

int BatchNorm::forward_inplace(Mat& bottom_top_blob) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float a = a_data[q];
        float b = b_data[q];

        for (int i=0; i<size; i++)
        {
            ptr[i] = b * ptr[i] + a;
        }
    }

    return 0;
}

} // namespace ncnn

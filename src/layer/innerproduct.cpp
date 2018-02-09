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

#include "innerproduct.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct)

InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;
}

int InnerProduct::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);

    return 0;
}

int InnerProduct::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int InnerProduct::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(num_output);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for
    for (int p=0; p<num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* w = (const float*)weight_data + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        top_blob[p] = sum;
    }

    return 0;
}

} // namespace ncnn

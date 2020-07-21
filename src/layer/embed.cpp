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

#include "embed.h"

#include <string.h>

namespace ncnn {

Embed::Embed()
{
    one_blob_only = true;
    support_inplace = false;
}

int Embed::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    input_dim = pd.get(1, 0);
    bias_term = pd.get(2, 0);
    weight_data_size = pd.get(3, 0);

    return 0;
}

int Embed::load_model(const ModelBin& mb)
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

int Embed::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int words = static_cast<int>(bottom_blob.total());

    top_blob.create(num_output, words, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < words; q++)
    {
        float* outptr = top_blob.row(q);

        int word_index = ((const int*)bottom_blob)[q];

        if (word_index < 0)
            word_index = 0;
        if (word_index >= input_dim)
            word_index = input_dim - 1;

        const float* em = (const float*)weight_data + num_output * word_index;

        memcpy(outptr, em, num_output * sizeof(float));

        if (bias_term)
        {
            for (int p = 0; p < num_output; p++)
            {
                outptr[p] += bias_data[p];
            }
        }
    }

    return 0;
}

} // namespace ncnn

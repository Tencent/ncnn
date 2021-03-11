// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "gelu.h"

#include <math.h>

namespace ncnn {

GELU::GELU()
{
    one_blob_only = true;
    support_inplace = true;
}

int GELU::load_param(const ParamDict& pd)
{
    fast_gelu = pd.get(0, 0) != 0;

    return 0;
}

int GELU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (fast_gelu)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
                ptr[i] = 0.5f * ptr[i] * (1.0f + tanhf(0.79788452f * (ptr[i] + 0.044715f * ptr[i] * ptr[i] * ptr[i])));
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                // y = x * P(X <= x) where X ~ N(0, 1)
                ptr[i] = 0.5f * ptr[i] * erfcf(-0.70710678f * ptr[i]);
            }
        }
    }

    return 0;
}

} // namespace ncnn

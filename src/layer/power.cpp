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

#include "power.h"

#include <math.h>

namespace ncnn {

Power::Power()
{
    one_blob_only = true;
    support_inplace = true;
}

int Power::load_param(const ParamDict& pd)
{
    power = pd.get(0, 1.f);
    scale = pd.get(1, 1.f);
    shift = pd.get(2, 0.f);

    return 0;
}

int Power::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = static_cast<float>(pow((shift + ptr[i] * scale), power));
        }
    }

    return 0;
}

} // namespace ncnn

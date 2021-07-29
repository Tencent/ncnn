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

#include "l2normalization.h"

namespace ncnn {

L2Normalization::L2Normalization()
{
    one_blob_only = true;
    support_inplace = true;
}

int L2Normalization::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f);

    return 0;
}

int L2Normalization::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    int input_data_size = size * channels;
    float sq_l2_norm = 0;

    for ( int j = 0; j < channels; j++)
    {
        float* ptr = bottom_top_blob.channel(j);
        const float val = ptr[0] * ptr[0];
        sq_l2_norm += val;
    }
    const float l2_norm = sqrt(sq_l2_norm);
    for ( int j = 0; j < channels; j++)
    {
        float* ptr = bottom_top_blob.channel(j);
        ptr[0] = ptr[0] / l2_norm;
    }    

    return 0;
}

} // namespace ncnn

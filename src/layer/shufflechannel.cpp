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

#include "shufflechannel.h"

namespace ncnn {

ShuffleChannel::ShuffleChannel()
{
    one_blob_only = true;
    support_inplace = false;
}

int ShuffleChannel::load_param(const ParamDict& pd)
{
    group = pd.get(0, 1);
    reverse = pd.get(1, 0);

    return 0;
}

int ShuffleChannel::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0)
    {
        // reject invalid group
        return -100;
    }

    int _group = reverse ? channels / group : group;
    int channels_per_group = channels / _group;

    top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const size_t feature_sz = (size_t)w * h * elemsize;
    for (int i = 0; i < _group; i++)
    {
        for (int j = 0; j < channels_per_group; j++)
        {
            int src_q = channels_per_group * i + j;
            int dst_q = _group * j + i;
            memcpy(top_blob.channel(dst_q), bottom_blob.channel(src_q), feature_sz);
        }
    }

    return 0;
}

} // namespace ncnn

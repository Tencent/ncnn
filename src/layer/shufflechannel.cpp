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

DEFINE_LAYER_CREATOR(ShuffleChannel)

ShuffleChannel::ShuffleChannel()
{
}

ShuffleChannel::~ShuffleChannel()
{
}


int ShuffleChannel::load_param(const ParamDict &pd)
{
    group = pd.get(0, 1);
    return 0;
}

int ShuffleChannel::forward(const Mat &bottom_blob, Mat &top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int c = bottom_blob.c;
    int cstep = bottom_blob.cstep;
    int chs_per_group = c / group;

    if (c != chs_per_group * group) {
//        cout << "Wrong group num";
//        error;
        // reject invalid group
        return -100;
    }

    top_blob.create(w, h, c);
    if (top_blob.empty())
        return -100;

    float *top_dat = top_blob.data;
    float *bot_dat = bottom_blob.data;
    int q;
    for (int i = 0; i != group; ++i) {
        for (int j = 0; j != chs_per_group; ++j) {
            q = chs_per_group * i + j;
            float *dst_ch = top_dat + cstep * q;
            q = chs_per_group * j + i;
            float *src_ch = bot_dat + cstep * q;
            memcpy(dst_ch, src_ch, cstep); // or w * h
        }
    }
    return 0;
}

} // namespace ncnn

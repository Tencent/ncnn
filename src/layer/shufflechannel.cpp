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

int ShuffleChannel::forward(const Mat &bottom_blob, Mat &top_blob) const
{
    int c = bottom_blob.c;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
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

    int dst_q;
    int src_q;
    // cstep * sizeof(float) if addr aligned needed
    size_t feature_sz = w * h * sizeof(float);
    for (int i = 0; i != group; ++i) {
        for (int j = 0; j != chs_per_group; ++j) {
            src_q = chs_per_group * i + j;
            dst_q = group * j + i;
            memcpy(top_blob.channel(dst_q), bottom_blob.channel(src_q),
                   feature_sz);
        }
    }
    return 0;
}

} // namespace ncnn

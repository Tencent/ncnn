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

#include "squeeze.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Squeeze)

Squeeze::Squeeze()
{
    one_blob_only = true;
    support_inplace = false;
}

int Squeeze::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    if (channels == 1)
    {
        if (h == 1)
            top_blob = bottom_blob.reshape(w);
        else
            top_blob = bottom_blob.reshape(w, h);
    }
    else if (h == 1)
    {
        if (w == 1)
            top_blob = bottom_blob.reshape(channels);
        else
            top_blob = bottom_blob.reshape(w, channels);
    }
    else if (w == 1)
    {
        if (h == 1)
            top_blob = bottom_blob.reshape(channels);
        else
            top_blob = bottom_blob.reshape(h, channels);
    }
    else
        top_blob = bottom_blob;

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn

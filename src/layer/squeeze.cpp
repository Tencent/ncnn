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

int Squeeze::load_param(const ParamDict& pd)
{
    squeeze_w = pd.get(0, 0);
    squeeze_h = pd.get(1, 0);
    squeeze_c = pd.get(2, 0);

    return 0;
}

int Squeeze::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    top_blob = bottom_blob;

    if (squeeze_c && dims == 3 && channels == 1)
    {
        if (squeeze_h && h == 1)
            top_blob = bottom_blob.reshape(w, opt.blob_allocator);
        else
            top_blob = bottom_blob.reshape(w, h, opt.blob_allocator);
    }
    else if (squeeze_h && dims >= 2 && h == 1)
    {
        if (squeeze_w && w == 1)
            top_blob = bottom_blob.reshape(channels, opt.blob_allocator);
        else
            top_blob = bottom_blob.reshape(w, channels, opt.blob_allocator);
    }
    else if (squeeze_w && dims >= 1 && w == 1)
    {
        if (squeeze_h && h == 1)
            top_blob = bottom_blob.reshape(channels, opt.blob_allocator);
        else
            top_blob = bottom_blob.reshape(h, channels, opt.blob_allocator);
    }

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn

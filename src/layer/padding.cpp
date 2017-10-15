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

#include "padding.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Padding)

Padding::Padding()
{
    one_blob_only = true;
    support_inplace = false;
}

int Padding::load_param(const ParamDict& pd)
{
    top = pd.get(0, 0);
    bottom = pd.get(1, 0);
    left = pd.get(2, 0);
    right = pd.get(3, 0);
    type = pd.get(4, 0);
    value = pd.get(5, 0.f);

    return 0;
}

int Padding::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    copy_make_border(bottom_blob, top_blob, top, bottom, left, right, type, value);

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn

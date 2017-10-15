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

#include "crop.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Crop)

Crop::Crop()
{
}

int Crop::load_param(const ParamDict& pd)
{
    woffset = pd.get(0, 0);
    hoffset = pd.get(1, 0);

    return 0;
}

int Crop::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    int top = hoffset;
    int bottom = h - outh - hoffset;
    int left = woffset;
    int right = w - outw - woffset;

    Mat& top_blob = top_blobs[0];

    copy_cut_border(bottom_blob, top_blob, top, bottom, left, right);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn

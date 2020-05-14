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

#ifndef LAYER_PRIORBOX_H
#define LAYER_PRIORBOX_H

#include "layer.h"

namespace ncnn {

class PriorBox : public Layer
{
public:
    PriorBox();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    Mat min_sizes;
    Mat max_sizes;
    Mat aspect_ratios;
    float variances[4];
    int flip;
    int clip;
    int image_width;
    int image_height;
    float step_width;
    float step_height;
    float offset;
    bool step_mmdetection;
    bool center_mmdetection;
};

} // namespace ncnn

#endif // LAYER_PRIORBOX_H

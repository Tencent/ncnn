// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_GRIDSAMPLE_H
#define LAYER_GRIDSAMPLE_H

#include "layer.h"

namespace ncnn {

class GridSample : public Layer
{
public:
    GridSample();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    enum InterpolationMode // 1=bilinear  2=nearest  3=bicubic
    {
        Interpolation_BILINEAR = 1,
        Interpolation_NEAREST = 2,
        Interpolation_BICUBIC = 3
    };

    enum PaddingMode // 1=zeros     2=border   3=reflection
    {
        Padding_ZEROS = 1,
        Padding_BORDER = 2,
        Padding_REFLECTION = 3
    };

public:
    // param
    int sample_type;  // 1=bilinear  2=nearest  3=bicubic
    int padding_mode; // 1=zeros     2=border   3=reflection
    int align_corner;

    int permute_fusion;
};

} // namespace ncnn

#endif // LAYER_GRIDSAMPLE_H

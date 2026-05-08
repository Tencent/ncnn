// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

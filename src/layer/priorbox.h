// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

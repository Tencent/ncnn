// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROIPOOLING_H
#define LAYER_ROIPOOLING_H

#include "layer.h"

namespace ncnn {

class ROIPooling : public Layer
{
public:
    ROIPooling();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int pooled_width;
    int pooled_height;
    float spatial_scale;
};

} // namespace ncnn

#endif // LAYER_ROIPOOLING_H

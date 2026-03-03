// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PSROIPOOLING_H
#define LAYER_PSROIPOOLING_H

#include "layer.h"

namespace ncnn {

class PSROIPooling : public Layer
{
public:
    PSROIPooling();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int pooled_width;
    int pooled_height;
    float spatial_scale;
    int output_dim;
};

} // namespace ncnn

#endif // LAYER_PSROIPOOLING_H

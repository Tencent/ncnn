// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWIGLU_H
#define LAYER_SWIGLU_H

#include "layer.h"

namespace ncnn {

class SwiGLU : public Layer
{
public:
    SwiGLU();

    virtual int load_param(const ParamDict& pd);

    using Layer::forward;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    float sigmoid_scale;
    float up_offset;
    float activation_limit;
};

} // namespace ncnn

#endif // LAYER_SWIGLU_H

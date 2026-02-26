// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RELU_H
#define LAYER_RELU_H

#include "layer.h"

namespace ncnn {

class ReLU : public Layer
{
public:
    ReLU();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float slope;
};

} // namespace ncnn

#endif // LAYER_RELU_H

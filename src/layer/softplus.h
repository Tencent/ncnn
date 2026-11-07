// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTPLUS_H
#define LAYER_SOFTPLUS_H

#include "layer.h"

namespace ncnn {

class Softplus : public Layer
{
public:
    Softplus();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float threshold;
    int has_threshold;
};

} // namespace ncnn

#endif // LAYER_SOFTPLUS_H

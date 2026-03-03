// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POWER_H
#define LAYER_POWER_H

#include "layer.h"

namespace ncnn {

class Power : public Layer
{
public:
    Power();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float power;
    float scale;
    float shift;
};

} // namespace ncnn

#endif // LAYER_POWER_H

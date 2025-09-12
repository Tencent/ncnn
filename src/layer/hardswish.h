// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSWISH_H
#define LAYER_HARDSWISH_H

#include "layer.h"

namespace ncnn {

class HardSwish : public Layer
{
public:
    HardSwish();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float alpha, beta, lower, upper;
};

} // namespace ncnn

#endif // LAYER_HARDSWISH_H

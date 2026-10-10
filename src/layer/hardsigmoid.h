// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSIGMOID_H
#define LAYER_HARDSIGMOID_H

#include "layer.h"

namespace ncnn {

class HardSigmoid : public Layer
{
public:
    HardSigmoid();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float alpha, beta, lower, upper;
};

} // namespace ncnn

#endif // LAYER_HARDSIGMOID_H

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EXP_H
#define LAYER_EXP_H

#include "layer.h"

namespace ncnn {

class Exp : public Layer
{
public:
    Exp();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float base;
    float scale;
    float shift;
};

} // namespace ncnn

#endif // LAYER_EXP_H

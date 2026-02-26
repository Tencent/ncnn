// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PERMUTE_H
#define LAYER_PERMUTE_H

#include "layer.h"

namespace ncnn {

class Permute : public Layer
{
public:
    Permute();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int order_type;
};

} // namespace ncnn

#endif // LAYER_PERMUTE_H

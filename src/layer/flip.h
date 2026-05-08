// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLIP_H
#define LAYER_FLIP_H

#include "layer.h"

namespace ncnn {

class Flip : public Layer
{
public:
    Flip();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    Mat axes;
};

} // namespace ncnn

#endif // LAYER_FLIP_H

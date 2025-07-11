// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SIGMOID_H
#define LAYER_SIGMOID_H

#include "layer.h"

namespace ncnn {

class Sigmoid : public Layer
{
public:
    Sigmoid();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SIGMOID_H

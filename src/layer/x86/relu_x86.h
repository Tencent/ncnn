// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RELU_X86_H
#define LAYER_RELU_X86_H

#include "relu.h"

namespace ncnn {

class ReLU_x86 : public ReLU
{
public:
    ReLU_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RELU_X86_H

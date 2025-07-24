// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RELU_MIPS_H
#define LAYER_RELU_MIPS_H

#include "relu.h"

namespace ncnn {

class ReLU_mips : public ReLU
{
public:
    ReLU_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RELU_MIPS_H

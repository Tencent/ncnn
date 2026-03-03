// Copyright 2020 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SIGMOID_MIPS_H
#define LAYER_SIGMOID_MIPS_H

#include "sigmoid.h"

namespace ncnn {

class Sigmoid_mips : public Sigmoid
{
public:
    Sigmoid_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SIGMOID_MIPS_H

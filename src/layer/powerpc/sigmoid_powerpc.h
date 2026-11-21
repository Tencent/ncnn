// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SIGMOID_POWERPC_H
#define LAYER_SIGMOID_POWERPC_H

#include "sigmoid.h"

namespace ncnn {

class Sigmoid_powerpc : public Sigmoid
{
public:
    Sigmoid_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SIGMOID_POWERPC_H

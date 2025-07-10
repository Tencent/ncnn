// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SIGMOID_RISCV_H
#define LAYER_SIGMOID_RISCV_H

#include "sigmoid.h"

namespace ncnn {

class Sigmoid_riscv : public Sigmoid
{
public:
    Sigmoid_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SIGMOID_RISCV_H

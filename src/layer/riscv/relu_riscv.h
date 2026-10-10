// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RELU_RISCV_H
#define LAYER_RELU_RISCV_H

#include "relu.h"

namespace ncnn {

class ReLU_riscv : public ReLU
{
public:
    ReLU_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_RELU_RISCV_H

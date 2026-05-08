// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTMAX_RISCV_H
#define LAYER_SOFTMAX_RISCV_H

#include "softmax.h"

namespace ncnn {

class Softmax_riscv : public Softmax
{
public:
    Softmax_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_RISCV_H

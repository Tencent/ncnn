// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DROPOUT_RISCV_H
#define LAYER_DROPOUT_RISCV_H

#include "dropout.h"

namespace ncnn {

class Dropout_riscv : public Dropout
{
public:
    Dropout_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DROPOUT_RISCV_H

// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SELU_RISCV_H
#define LAYER_SELU_RISCV_H

#include "selu.h"

namespace ncnn {

class SELU_riscv : public SELU
{
public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SELU_RISCV_H

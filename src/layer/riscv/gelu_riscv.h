// Copyright 2022 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GELU_RISCV_H
#define LAYER_GELU_RISCV_H

#include "gelu.h"

namespace ncnn {

class GELU_riscv : public GELU
{
public:
    GELU_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GELU_RISCV_H

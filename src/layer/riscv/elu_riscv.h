// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELU_RISCV_H
#define LAYER_ELU_RISCV_H

#include "elu.h"

namespace ncnn {

class ELU_riscv : public ELU
{
public:
    ELU_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ELU_RISCV_H

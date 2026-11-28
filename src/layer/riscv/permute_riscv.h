// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PERMUTE_RISCV_H
#define LAYER_PERMUTE_RISCV_H

#include "permute.h"

namespace ncnn {

class Permute_riscv : public Permute
{
public:
    Permute_riscv();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PERMUTE_RISCV_H

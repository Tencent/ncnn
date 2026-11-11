// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GROUPNORM_RISCV_H
#define LAYER_GROUPNORM_RISCV_H

#include "groupnorm.h"

namespace ncnn {

class GroupNorm_riscv : public GroupNorm
{
public:
    GroupNorm_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GROUPNORM_RISCV_H

// Copyright 2022 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INSTANCENORM_RISCV_H
#define LAYER_INSTANCENORM_RISCV_H

#include "instancenorm.h"

namespace ncnn {
class InstanceNorm_riscv : public InstanceNorm
{
public:
    InstanceNorm_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
#endif
};
} // namespace ncnn

#endif // LAYER_INSTANCENORM_RISCV_H

// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSIGMOID_RISCV_H
#define LAYER_HARDSIGMOID_RISCV_H

#include "hardsigmoid.h"

namespace ncnn {

class HardSigmoid_riscv : public HardSigmoid
{
public:
    HardSigmoid_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_HARDSIGMOID_RISCV_H

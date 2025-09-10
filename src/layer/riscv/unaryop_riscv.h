// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNARYOP_RISCV_H
#define LAYER_UNARYOP_RISCV_H

#include "unaryop.h"

namespace ncnn {

class UnaryOp_riscv : public UnaryOp
{
public:
    UnaryOp_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_UNARYOP_RISCV_H

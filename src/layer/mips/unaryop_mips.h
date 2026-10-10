// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNARYOP_MIPS_H
#define LAYER_UNARYOP_MIPS_H

#include "unaryop.h"

namespace ncnn {

class UnaryOp_mips : public UnaryOp
{
public:
    UnaryOp_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_UNARYOP_MIPS_H

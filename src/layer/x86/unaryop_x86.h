// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNARYOP_X86_H
#define LAYER_UNARYOP_X86_H

#include "unaryop.h"

namespace ncnn {

class UnaryOp_x86 : public UnaryOp
{
public:
    UnaryOp_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_UNARYOP_X86_H

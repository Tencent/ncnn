// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNARYOP_POWERPC_H
#define LAYER_UNARYOP_POWERPC_H

#include "unaryop.h"

namespace ncnn {

class UnaryOp_powerpc : public UnaryOp
{
public:
    UnaryOp_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_UNARYOP_POWERPC_H

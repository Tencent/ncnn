// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EXP_POWERPC_H
#define LAYER_EXP_POWERPC_H

#include "exp.h"

namespace ncnn {

class Exp_powerpc : public Exp
{
public:
    Exp_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_EXP_POWERPC_H

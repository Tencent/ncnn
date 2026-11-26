// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSIGMOID_POWERPC_H
#define LAYER_HARDSIGMOID_POWERPC_H

#include "hardsigmoid.h"

namespace ncnn {

class HardSigmoid_powerpc : public HardSigmoid
{
public:
    HardSigmoid_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSIGMOID_POWERPC_H

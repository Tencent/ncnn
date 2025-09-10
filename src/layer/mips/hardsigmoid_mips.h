// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSIGMOID_MIPS_H
#define LAYER_HARDSIGMOID_MIPS_H

#include "hardsigmoid.h"

namespace ncnn {

class HardSigmoid_mips : public HardSigmoid
{
public:
    HardSigmoid_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSIGMOID_MIPS_H

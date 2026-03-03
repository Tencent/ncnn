// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSIGMOID_X86_H
#define LAYER_HARDSIGMOID_X86_H

#include "hardsigmoid.h"

namespace ncnn {

class HardSigmoid_x86 : public HardSigmoid
{
public:
    HardSigmoid_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSIGMOID_X86_H

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSWISH_X86_H
#define LAYER_HARDSWISH_X86_H

#include "hardswish.h"

namespace ncnn {

class HardSwish_x86 : public HardSwish
{
public:
    HardSwish_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSWISH_X86_H

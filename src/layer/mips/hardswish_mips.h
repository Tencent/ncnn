// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSWISH_MIPS_H
#define LAYER_HARDSWISH_MIPS_H

#include "hardswish.h"

namespace ncnn {

class HardSwish_mips : public HardSwish
{
public:
    HardSwish_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSWISH_MIPS_H

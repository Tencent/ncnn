// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSWISH_POWERPC_H
#define LAYER_HARDSWISH_POWERPC_H

#include "hardswish.h"

namespace ncnn {

class HardSwish_powerpc : public HardSwish
{
public:
    HardSwish_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSWISH_POWERPC_H

// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MISH_MIPS_H
#define LAYER_MISH_MIPS_H

#include "mish.h"

namespace ncnn {

class Mish_mips : public Mish
{
public:
    Mish_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MISH_MIPS_H

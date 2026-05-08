// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWISH_MIPS_H
#define LAYER_SWISH_MIPS_H

#include "swish.h"

namespace ncnn {

class Swish_mips : public Swish
{
public:
    Swish_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SWISH_MIPS_H

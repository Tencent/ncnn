// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MISH_X86_H
#define LAYER_MISH_X86_H

#include "mish.h"

namespace ncnn {

class Mish_x86 : public Mish
{
public:
    Mish_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MISH_X86_H

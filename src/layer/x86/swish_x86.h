// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWISH_X86_H
#define LAYER_SWISH_X86_H

#include "swish.h"

namespace ncnn {

class Swish_x86 : public Swish
{
public:
    Swish_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SWISH_X86_H

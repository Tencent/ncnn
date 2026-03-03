// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTMAX_X86_H
#define LAYER_SOFTMAX_X86_H

#include "softmax.h"

namespace ncnn {

class Softmax_x86 : public Softmax
{
public:
    Softmax_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_X86_H

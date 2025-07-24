// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REQUANTIZE_X86_H
#define LAYER_REQUANTIZE_X86_H

#include "requantize.h"

namespace ncnn {

class Requantize_x86 : public Requantize
{
public:
    Requantize_x86();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REQUANTIZE_X86_H

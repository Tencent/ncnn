// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_QUANTIZE_X86_H
#define LAYER_QUANTIZE_X86_H

#include "quantize.h"

namespace ncnn {

class Quantize_x86 : public Quantize
{
public:
    Quantize_x86();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_QUANTIZE_X86_H

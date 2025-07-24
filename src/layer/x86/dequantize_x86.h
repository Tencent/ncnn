// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEQUANTIZE_X86_H
#define LAYER_DEQUANTIZE_X86_H

#include "dequantize.h"

namespace ncnn {

class Dequantize_x86 : public Dequantize
{
public:
    Dequantize_x86();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DEQUANTIZE_X86_H

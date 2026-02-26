// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEQUANTIZE_MIPS_H
#define LAYER_DEQUANTIZE_MIPS_H

#include "dequantize.h"

namespace ncnn {

class Dequantize_mips : public Dequantize
{
public:
    Dequantize_mips();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DEQUANTIZE_MIPS_H

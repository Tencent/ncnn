// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_QUANTIZE_MIPS_H
#define LAYER_QUANTIZE_MIPS_H

#include "quantize.h"

namespace ncnn {

class Quantize_mips : public Quantize
{
public:
    Quantize_mips();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_QUANTIZE_MIPS_H

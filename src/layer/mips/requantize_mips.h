// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REQUANTIZE_MIPS_H
#define LAYER_REQUANTIZE_MIPS_H

#include "requantize.h"

namespace ncnn {

class Requantize_mips : public Requantize
{
public:
    Requantize_mips();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REQUANTIZE_MIPS_H

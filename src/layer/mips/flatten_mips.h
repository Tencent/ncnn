// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLATTEN_MIPS_H
#define LAYER_FLATTEN_MIPS_H

#include "flatten.h"

namespace ncnn {

class Flatten_mips : public Flatten
{
public:
    Flatten_mips();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_FLATTEN_MIPS_H

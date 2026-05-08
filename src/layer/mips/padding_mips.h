// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PADDING_MIPS_H
#define LAYER_PADDING_MIPS_H

#include "padding.h"

namespace ncnn {

class Padding_mips : public Padding
{
public:
    Padding_mips();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PADDING_MIPS_H

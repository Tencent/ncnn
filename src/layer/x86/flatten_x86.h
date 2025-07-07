// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLATTEN_X86_H
#define LAYER_FLATTEN_X86_H

#include "flatten.h"

namespace ncnn {

class Flatten_x86 : public Flatten
{
public:
    Flatten_x86();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_FLATTEN_X86_H

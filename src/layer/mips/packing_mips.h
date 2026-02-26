// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PACKING_MIPS_H
#define LAYER_PACKING_MIPS_H

#include "packing.h"

namespace ncnn {

class Packing_mips : public Packing
{
public:
    Packing_mips();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PACKING_MIPS_H

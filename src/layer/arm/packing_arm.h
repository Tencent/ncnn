// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PACKING_ARM_H
#define LAYER_PACKING_ARM_H

#include "packing.h"

namespace ncnn {

class Packing_arm : public Packing
{
public:
    Packing_arm();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PACKING_ARM_H

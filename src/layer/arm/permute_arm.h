// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PERMUTE_ARM_H
#define LAYER_PERMUTE_ARM_H

#include "permute.h"

namespace ncnn {

class Permute_arm : public Permute
{
public:
    Permute_arm();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    int forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PERMUTE_ARM_H

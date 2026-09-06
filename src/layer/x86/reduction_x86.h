// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REDUCTION_X86_H
#define LAYER_REDUCTION_X86_H

#include "reduction.h"

namespace ncnn {

class Reduction_x86 : public Reduction
{
public:
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REDUCTION_X86_H

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REDUCTION_POWERPC_H
#define LAYER_REDUCTION_POWERPC_H

#include "reduction.h"

namespace ncnn {

class Reduction_powerpc : public Reduction
{
public:
    Reduction_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REDUCTION_POWERPC_H

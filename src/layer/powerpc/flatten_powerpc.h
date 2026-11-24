// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLATTEN_POWERPC_H
#define LAYER_FLATTEN_POWERPC_H

#include "flatten.h"

namespace ncnn {

class Flatten_powerpc : public Flatten
{
public:
    Flatten_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_FLATTEN_POWERPC_H

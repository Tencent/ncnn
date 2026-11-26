// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POOLING1D_POWERPC_H
#define LAYER_POOLING1D_POWERPC_H

#include "pooling1d.h"

namespace ncnn {

class Pooling1D_powerpc : public Pooling1D
{
public:
    Pooling1D_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_POOLING1D_POWERPC_H

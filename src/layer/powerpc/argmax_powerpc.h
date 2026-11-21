// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ARGMAX_POWERPC_H
#define LAYER_ARGMAX_POWERPC_H

#include "argmax.h"

namespace ncnn {

class ArgMax_powerpc : public ArgMax
{
public:
    ArgMax_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ARGMAX_POWERPC_H

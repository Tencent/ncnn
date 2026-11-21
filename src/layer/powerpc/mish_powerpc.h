// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MISH_POWERPC_H
#define LAYER_MISH_POWERPC_H

#include "mish.h"

namespace ncnn {

class Mish_powerpc : public Mish
{
public:
    Mish_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MISH_POWERPC_H

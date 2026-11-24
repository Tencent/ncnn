// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWISH_POWERPC_H
#define LAYER_SWISH_POWERPC_H

#include "swish.h"

namespace ncnn {

class Swish_powerpc : public Swish
{
public:
    Swish_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SWISH_POWERPC_H

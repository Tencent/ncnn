// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TANH_POWERPC_H
#define LAYER_TANH_POWERPC_H

#include "tanh.h"

namespace ncnn {

class TanH_powerpc : public TanH
{
public:
    TanH_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_TANH_POWERPC_H

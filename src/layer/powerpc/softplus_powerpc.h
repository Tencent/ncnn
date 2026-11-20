// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTPLUS_POWERPC_H
#define LAYER_SOFTPLUS_POWERPC_H

#include "softplus.h"

namespace ncnn {

class Softplus_powerpc : public Softplus
{
public:
    Softplus_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SOFTPLUS_POWERPC_H

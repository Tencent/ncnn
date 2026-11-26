// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GELU_POWERPC_H
#define LAYER_GELU_POWERPC_H

#include "gelu.h"

namespace ncnn {

class GELU_powerpc : public GELU
{
public:
    GELU_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GELU_POWERPC_H

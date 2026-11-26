// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SELU_POWERPC_H
#define LAYER_SELU_POWERPC_H

#include "selu.h"

namespace ncnn {

class SELU_powerpc : public SELU
{
public:
    SELU_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SELU_POWERPC_H

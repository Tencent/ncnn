// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CELU_POWERPC_H
#define LAYER_CELU_POWERPC_H

#include "celu.h"

namespace ncnn {

class CELU_powerpc : public CELU
{
public:
    CELU_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CELU_POWERPC_H

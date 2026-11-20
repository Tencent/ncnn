// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELU_POWERPC_H
#define LAYER_ELU_POWERPC_H

#include "elu.h"

namespace ncnn {

class ELU_powerpc : public ELU
{
public:
    ELU_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ELU_POWERPC_H

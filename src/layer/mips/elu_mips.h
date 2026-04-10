// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELU_MIPS_H
#define LAYER_ELU_MIPS_H

#include "elu.h"

namespace ncnn {

class ELU_mips : public ELU
{
public:
    ELU_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_BF16
    virtual int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ELU_MIPS_H

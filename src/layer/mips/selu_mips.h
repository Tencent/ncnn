// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SELU_MIPS_H
#define LAYER_SELU_MIPS_H

#include "selu.h"

namespace ncnn {

class SELU_mips : public SELU
{
public:
    SELU_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    virtual int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SELU_MIPS_H

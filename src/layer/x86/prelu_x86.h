// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PRELU_X86_H
#define LAYER_PRELU_X86_H

#include "prelu.h"

namespace ncnn {

class PReLU_x86 : public PReLU
{
public:
    PReLU_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_PRELU_X86_H

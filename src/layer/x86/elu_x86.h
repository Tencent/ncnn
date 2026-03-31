// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELU_X86_H
#define LAYER_ELU_X86_H

#include "elu.h"

namespace ncnn {

class ELU_x86 : public ELU
{
public:
    ELU_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ELU_X86_H

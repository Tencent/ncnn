// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTPLUS_X86_H
#define LAYER_SOFTPLUS_X86_H

#include "softplus.h"

namespace ncnn {

class Softplus_x86 : public Softplus
{
public:
    Softplus_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SOFTPLUS_X86_H

// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INSTANCENORM_X86_H
#define LAYER_INSTANCENORM_X86_H

#include "instancenorm.h"

namespace ncnn {

class InstanceNorm_x86 : public InstanceNorm
{
public:
    InstanceNorm_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_INSTANCENORM_X86_H

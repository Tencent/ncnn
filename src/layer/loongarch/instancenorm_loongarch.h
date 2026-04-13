// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INSTANCENORM_LOONGARCH_H
#define LAYER_INSTANCENORM_LOONGARCH_H

#include "instancenorm.h"

namespace ncnn {

class InstanceNorm_loongarch : public InstanceNorm
{
public:
    InstanceNorm_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_BF16
    virtual int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_INSTANCENORM_LOONGARCH_H

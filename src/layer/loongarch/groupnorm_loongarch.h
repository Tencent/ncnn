// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GROUPNORM_LOONGARCH_H
#define LAYER_GROUPNORM_LOONGARCH_H

#include "groupnorm.h"

namespace ncnn {

class GroupNorm_loongarch : public GroupNorm
{
public:
    GroupNorm_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_BF16
    virtual int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_GROUPNORM_LOONGARCH_H

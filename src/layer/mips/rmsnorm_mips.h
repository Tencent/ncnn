// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RMSNORM_MIPS_H
#define LAYER_RMSNORM_MIPS_H

#include "rmsnorm.h"

namespace ncnn {

class RMSNorm_mips : public RMSNorm
{
public:
    RMSNorm_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_RMSNORM_MIPS_H

// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LAYERNORM_RISCV_H
#define LAYER_LAYERNORM_RISCV_H

#include "layernorm.h"

namespace ncnn {

class LayerNorm_riscv : public LayerNorm
{
public:
    LayerNorm_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_RISCV_H

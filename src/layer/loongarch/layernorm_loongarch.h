// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LAYERNORM_LOONGARCH_H
#define LAYER_LAYERNORM_LOONGARCH_H

#include "layernorm.h"

namespace ncnn {

class LayerNorm_loongarch : public LayerNorm
{
public:
    LayerNorm_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_BF16
    virtual int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_LOONGARCH_H

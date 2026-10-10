// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LAYERNORM_ARM_H
#define LAYER_LAYERNORM_ARM_H

#include "layernorm.h"

namespace ncnn {

class LayerNorm_arm : public LayerNorm
{
public:
    LayerNorm_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_ARM_H

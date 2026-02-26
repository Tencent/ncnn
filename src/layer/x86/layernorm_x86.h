// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LAYERNORM_X86_H
#define LAYER_LAYERNORM_X86_H

#include "layernorm.h"

namespace ncnn {

class LayerNorm_x86 : public LayerNorm
{
public:
    LayerNorm_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_X86_H

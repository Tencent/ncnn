// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LAYERNORM_POWERPC_H
#define LAYER_LAYERNORM_POWERPC_H

#include "layernorm.h"

namespace ncnn {

class LayerNorm_powerpc : public LayerNorm
{
public:
    LayerNorm_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_POWERPC_H

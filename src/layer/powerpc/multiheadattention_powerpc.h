// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MULTIHEADATTENTION_POWERPC_H
#define LAYER_MULTIHEADATTENTION_POWERPC_H

#include "multiheadattention.h"

namespace ncnn {

class MultiHeadAttention_powerpc : public MultiHeadAttention
{
public:
    MultiHeadAttention_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MULTIHEADATTENTION_POWERPC_H

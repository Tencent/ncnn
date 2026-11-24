// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EINSUM_POWERPC_H
#define LAYER_EINSUM_POWERPC_H

#include "einsum.h"

namespace ncnn {

class Einsum_powerpc : public Einsum
{
public:
    Einsum_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_EINSUM_POWERPC_H

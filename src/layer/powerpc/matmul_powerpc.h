// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MATMUL_POWERPC_H
#define LAYER_MATMUL_POWERPC_H

#include "matmul.h"

namespace ncnn {

class MatMul_powerpc : public MatMul
{
public:
    MatMul_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MATMUL_POWERPC_H

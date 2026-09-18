// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MATMUL_LOONGARCH_H
#define LAYER_MATMUL_LOONGARCH_H

#include "matmul.h"

namespace ncnn {

class MatMul_loongarch : public MatMul
{
public:
    MatMul_loongarch();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    Layer* gemm;
};

} // namespace ncnn

#endif // LAYER_MATMUL_LOONGARCH_H

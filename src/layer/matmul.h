// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MATMUL_H
#define LAYER_MATMUL_H

#include "layer.h"

namespace ncnn {

class MatMul : public Layer
{
public:
    MatMul();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int transB;
};

} // namespace ncnn

#endif // LAYER_MATMUL_H

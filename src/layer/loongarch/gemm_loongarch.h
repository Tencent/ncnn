// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_LOONGARCH_H
#define LAYER_GEMM_LOONGARCH_H

#include "gemm.h"

namespace ncnn {

class Gemm_loongarch : public Gemm
{
public:
    Gemm_loongarch();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GEMM_LOONGARCH_H

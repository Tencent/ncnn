// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_MIPS_H
#define LAYER_GEMM_MIPS_H

#include "gemm.h"

namespace ncnn {

class Gemm_mips : public Gemm
{
public:
    Gemm_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GEMM_MIPS_H

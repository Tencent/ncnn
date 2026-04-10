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

    virtual int create_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int nT;
    Mat AT_data;
    Mat BT_data;
    Mat CT_data;
};

} // namespace ncnn

#endif // LAYER_GEMM_MIPS_H

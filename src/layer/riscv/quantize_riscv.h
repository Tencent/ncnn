// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_QUANTIZE_RISCV_H
#define LAYER_QUANTIZE_RISCV_H

#include "quantize.h"

namespace ncnn {

// Ref: src/layer/arm/quantize_arm.cpp
class Quantize_riscv : public Quantize
{
public:
    Quantize_riscv();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_QUANTIZE_RISCV_H

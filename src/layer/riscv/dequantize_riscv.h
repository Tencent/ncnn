// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEQUANTIZE_RISCV_H
#define LAYER_DEQUANTIZE_RISCV_H

#include "dequantize.h"

namespace ncnn {

class Dequantize_riscv : public Dequantize
{
public:
    Dequantize_riscv();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_DEQUANTIZE_RISCV_H

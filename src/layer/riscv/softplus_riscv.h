// Copyright 2026 ihb2032 <hebome@foxmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTPLUS_RISCV_H
#define LAYER_SOFTPLUS_RISCV_H

#include "softplus.h"

namespace ncnn {

class Softplus_riscv : public Softplus
{
public:
    Softplus_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SOFTPLUS_RISCV_H

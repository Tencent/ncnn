// Copyright 2026 ihb2032 <hebome@foxmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHRINK_RISCV_H
#define LAYER_SHRINK_RISCV_H

#include "shrink.h"

namespace ncnn {

class Shrink_riscv : public Shrink
{
public:
    Shrink_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SHRINK_RISCV_H

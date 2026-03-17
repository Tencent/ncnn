// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ERF_RISCV_H
#define LAYER_ERF_RISCV_H

#include "erf.h"

namespace ncnn {

class Erf_riscv : public Erf
{
public:
    Erf_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ERF_RISCV_H

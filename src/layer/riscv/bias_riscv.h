// Copyright 2025 AtomAlpaca <atal@anche.no>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BIAS_RISCV_H
#define LAYER_BIAS_RISCV_H

#include "bias.h"

namespace ncnn {

class Bias_riscv : public Bias
{
public:
    Bias_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_BIAS_RISCV_H

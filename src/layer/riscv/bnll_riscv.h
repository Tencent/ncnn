// Copyright 2025 AtomAlpaca <atal@anche.no>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BNLL_RISCV_H
#define LAYER_BNLL_RISCV_H

#include "bnll.h"

namespace ncnn {

class BNLL_riscv : public BNLL
{
public:
    BNLL_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_BNLL_RISCV_H

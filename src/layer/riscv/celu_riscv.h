// Copyright 2025 AtomAlpaca <atal@anche.no>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CELU_RISCV_H
#define LAYER_CELU_RISCV_H

#include "celu.h"

namespace ncnn {

class CELU_riscv : public CELU
{
public:
    CELU_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_CELU_RISCV_H

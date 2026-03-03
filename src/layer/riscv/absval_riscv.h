// Copyright 2021 Xavier Hsinyuan <thelastlinex@hotmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ABSVAL_RISCV_H
#define LAYER_ABSVAL_RISCV_H

#include "absval.h"

namespace ncnn {

class AbsVal_riscv : public AbsVal
{
public:
    AbsVal_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ABSVAL_RISCV_H

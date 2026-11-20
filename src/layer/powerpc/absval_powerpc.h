// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ABSVAL_POWERPC_H
#define LAYER_ABSVAL_POWERPC_H

#include "absval.h"

namespace ncnn {

class AbsVal_powerpc : public AbsVal
{
public:
    AbsVal_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ABSVAL_POWERPC_H

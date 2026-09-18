// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LRN_MIPS_H
#define LAYER_LRN_MIPS_H

#include "lrn.h"

namespace ncnn {

class LRN_mips : public LRN
{
public:
    LRN_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_LRN_MIPS_H

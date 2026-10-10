// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LRN_LOONGARCH_H
#define LAYER_LRN_LOONGARCH_H

#include "lrn.h"

namespace ncnn {

class LRN_loongarch : public LRN
{
public:
    LRN_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_LRN_LOONGARCH_H

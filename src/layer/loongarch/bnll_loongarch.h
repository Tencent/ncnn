// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BNLL_LOONGARCH_H
#define LAYER_BNLL_LOONGARCH_H

#include "bnll.h"

namespace ncnn {

class BNLL_loongarch : public BNLL
{
public:
    BNLL_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_BNLL_LOONGARCH_H

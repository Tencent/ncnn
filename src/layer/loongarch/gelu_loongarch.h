// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GELU_LOONGARCH_H
#define LAYER_GELU_LOONGARCH_H

#include "gelu.h"

namespace ncnn {

class GELU_loongarch : public GELU
{
public:
    GELU_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_GELU_LOONGARCH_H

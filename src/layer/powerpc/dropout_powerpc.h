// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DROPOUT_POWERPC_H
#define LAYER_DROPOUT_POWERPC_H

#include "dropout.h"

namespace ncnn {

class Dropout_powerpc : public Dropout
{
public:
    Dropout_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_DROPOUT_POWERPC_H

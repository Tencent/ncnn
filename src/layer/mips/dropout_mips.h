// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DROPOUT_MIPS_H
#define LAYER_DROPOUT_MIPS_H

#include "dropout.h"

namespace ncnn {

class Dropout_mips : public Dropout
{
public:
    Dropout_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DROPOUT_MIPS_H

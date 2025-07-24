// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DROPOUT_ARM_H
#define LAYER_DROPOUT_ARM_H

#include "dropout.h"

namespace ncnn {

class Dropout_arm : public Dropout
{
public:
    Dropout_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DROPOUT_ARM_H

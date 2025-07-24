// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SELU_ARM_H
#define LAYER_SELU_ARM_H

#include "selu.h"

namespace ncnn {

class SELU_arm : public SELU
{
public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SELU_ARM_H

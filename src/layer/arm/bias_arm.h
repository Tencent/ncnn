// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BIAS_ARM_H
#define LAYER_BIAS_ARM_H

#include "bias.h"

namespace ncnn {

class Bias_arm : public Bias
{
public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BIAS_ARM_H

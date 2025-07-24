// Copyright 2019 BUG1989
// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REQUANTIZE_ARM_H
#define LAYER_REQUANTIZE_ARM_H

#include "requantize.h"

namespace ncnn {

class Requantize_arm : public Requantize
{
public:
    Requantize_arm();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REQUANTIZE_ARM_H

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ABSVAL_ARM_H
#define LAYER_ABSVAL_ARM_H

#include "absval.h"

namespace ncnn {

class AbsVal_arm : public AbsVal
{
public:
    AbsVal_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ABSVAL_ARM_H

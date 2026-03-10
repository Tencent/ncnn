// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELU_ARM_H
#define LAYER_ELU_ARM_H

#include "elu.h"

namespace ncnn {

class ELU_arm : public ELU
{
public:
    ELU_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ELU_ARM_H

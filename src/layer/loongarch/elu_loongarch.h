// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELU_LOONGARCH_H
#define LAYER_ELU_LOONGARCH_H

#include "elu.h"

namespace ncnn {

class ELU_loongarch : public ELU
{
public:
    ELU_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ELU_LOONGARCH_H

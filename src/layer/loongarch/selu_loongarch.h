// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SELU_LOONGARCH_H
#define LAYER_SELU_LOONGARCH_H

#include "selu.h"

namespace ncnn {

class SELU_loongarch : public SELU
{
public:
    SELU_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SELU_LOONGARCH_H

// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GELU_LOONGARCH_H
#define LAYER_GELU_LOONGARCH_H

#include "gelu.h"

namespace ncnn {

class GELU_loongarch : public GELU
{
public:
    GELU_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GELU_LOONGARCH_H

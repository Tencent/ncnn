// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSIGMOID_LOONGARCH_H
#define LAYER_HARDSIGMOID_LOONGARCH_H

#include "hardsigmoid.h"

namespace ncnn {

class HardSigmoid_loongarch : public HardSigmoid
{
public:
    HardSigmoid_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSIGMOID_LOONGARCH_H

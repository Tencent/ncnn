// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RELU_LOONGARCH_H
#define LAYER_RELU_LOONGARCH_H

#include "relu.h"

namespace ncnn {

class ReLU_loongarch : public ReLU
{
public:
    ReLU_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RELU_LOONGARCH_H

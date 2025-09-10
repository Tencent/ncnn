// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SIGMOID_LOONGARCH_H
#define LAYER_SIGMOID_LOONGARCH_H

#include "sigmoid.h"

namespace ncnn {

class Sigmoid_loongarch : public Sigmoid
{
public:
    Sigmoid_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SIGMOID_LOONGARCH_H

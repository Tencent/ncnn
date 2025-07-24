// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REQUANTIZE_LOONGARCH_H
#define LAYER_REQUANTIZE_LOONGARCH_H

#include "requantize.h"

namespace ncnn {

class Requantize_loongarch : public Requantize
{
public:
    Requantize_loongarch();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REQUANTIZE_LOONGARCH_H

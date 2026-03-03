// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DROPOUT_LOONGARCH_H
#define LAYER_DROPOUT_LOONGARCH_H

#include "dropout.h"

namespace ncnn {

class Dropout_loongarch : public Dropout
{
public:
    Dropout_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DROPOUT_LOONGARCH_H

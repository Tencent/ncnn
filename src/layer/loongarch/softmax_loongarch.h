// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTMAX_LOONGARCH_H
#define LAYER_SOFTMAX_LOONGARCH_H

#include "softmax.h"

namespace ncnn {

class Softmax_loongarch : public Softmax
{
public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_LOONGARCH_H

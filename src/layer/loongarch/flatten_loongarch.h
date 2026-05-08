// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLATTEN_LOONGARCH_H
#define LAYER_FLATTEN_LOONGARCH_H

#include "flatten.h"

namespace ncnn {

class Flatten_loongarch : public Flatten
{
public:
    Flatten_loongarch();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_FLATTEN_LOONGARCH_H

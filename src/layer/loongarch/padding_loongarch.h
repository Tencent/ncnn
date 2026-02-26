// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PADDING_LOONGARCH_H
#define LAYER_PADDING_LOONGARCH_H

#include "padding.h"

namespace ncnn {

class Padding_loongarch : public Padding
{
public:
    Padding_loongarch();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PADDING_LOONGARCH_H

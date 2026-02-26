// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MISH_LOONGARCH_H
#define LAYER_MISH_LOONGARCH_H

#include "mish.h"

namespace ncnn {

class Mish_loongarch : public Mish
{
public:
    Mish_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MISH_LOONGARCH_H

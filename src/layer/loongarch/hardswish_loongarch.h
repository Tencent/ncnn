// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSWISH_LOONGARCH_H
#define LAYER_HARDSWISH_LOONGARCH_H

#include "hardswish.h"

namespace ncnn {

class HardSwish_loongarch : public HardSwish
{
public:
    HardSwish_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSWISH_LOONGARCH_H

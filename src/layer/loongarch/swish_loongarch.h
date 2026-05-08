// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWISH_LOONGARCH_H
#define LAYER_SWISH_LOONGARCH_H

#include "swish.h"

namespace ncnn {

class Swish_loongarch : public Swish
{
public:
    Swish_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SWISH_LOONGARCH_H

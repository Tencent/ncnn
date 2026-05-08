// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TANH_LOONGARCH_H
#define LAYER_TANH_LOONGARCH_H

#include "tanh.h"

namespace ncnn {

class TanH_loongarch : public TanH
{
public:
    TanH_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_TANH_LOONGARCH_H

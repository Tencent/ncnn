// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PRELU_LOONGARCH_H
#define LAYER_PRELU_LOONGARCH_H

#include "prelu.h"

namespace ncnn {

class PReLU_loongarch : public PReLU
{
public:
    PReLU_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PRELU_LOONGARCH_H

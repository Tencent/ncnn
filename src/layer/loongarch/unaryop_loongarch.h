// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNARYOP_LOONGARCH_H
#define LAYER_UNARYOP_LOONGARCH_H

#include "unaryop.h"

namespace ncnn {

class UnaryOp_loongarch : public UnaryOp
{
public:
    UnaryOp_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_UNARYOP_LOONGARCH_H

// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELTWISE_LOONGARCH_H
#define LAYER_ELTWISE_LOONGARCH_H

#include "eltwise.h"

namespace ncnn {

class Eltwise_loongarch : public Eltwise
{
public:
    Eltwise_loongarch();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ELTWISE_LOONGARCH_H

// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POOLING_LOONGARCH_H
#define LAYER_POOLING_LOONGARCH_H

#include "pooling.h"

namespace ncnn {

class Pooling_loongarch : public Pooling
{
public:
    Pooling_loongarch();

    virtual int create_pipeline(const Option& opt);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_POOLING_LOONGARCH_H

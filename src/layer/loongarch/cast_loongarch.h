// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CAST_LOONGARCH_H
#define LAYER_CAST_LOONGARCH_H

#include "cast.h"

namespace ncnn {

class Cast_loongarch : public Cast
{
public:
    Cast_loongarch();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CAST_LOONGARCH_H

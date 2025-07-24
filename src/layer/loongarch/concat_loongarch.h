// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONCAT_LOONGARCH_H
#define LAYER_CONCAT_LOONGARCH_H

#include "concat.h"

namespace ncnn {

class Concat_loongarch : public Concat
{
public:
    Concat_loongarch();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CONCAT_LOONGARCH_H

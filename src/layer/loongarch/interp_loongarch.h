// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INTERP_LOONGARCH_H
#define LAYER_INTERP_LOONGARCH_H

#include "interp.h"

namespace ncnn {

class Interp_loongarch : public Interp
{
public:
    Interp_loongarch();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_INTERP_LOONGARCH_H

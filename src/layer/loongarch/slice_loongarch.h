// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SLICE_LOONGARCH_H
#define LAYER_SLICE_LOONGARCH_H

#include "slice.h"

namespace ncnn {

class Slice_loongarch : public Slice
{
public:
    Slice_loongarch();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SLICE_LOONGARCH_H

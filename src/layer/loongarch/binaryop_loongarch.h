// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BINARYOP_LOONGARCH_H
#define LAYER_BINARYOP_LOONGARCH_H

#include "binaryop.h"

namespace ncnn {

class BinaryOp_loongarch : public BinaryOp
{
public:
    BinaryOp_loongarch();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BINARYOP_LOONGARCH_H

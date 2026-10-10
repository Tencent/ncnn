// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_QUANTIZE_LOONGARCH_H
#define LAYER_QUANTIZE_LOONGARCH_H

#include "quantize.h"

namespace ncnn {

class Quantize_loongarch : public Quantize
{
public:
    Quantize_loongarch();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_QUANTIZE_LOONGARCH_H

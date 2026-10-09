// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BIAS_LOONGARCH_H
#define LAYER_BIAS_LOONGARCH_H

#include "bias.h"

namespace ncnn {

class Bias_loongarch : public Bias
{
public:
    Bias_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_BIAS_LOONGARCH_H

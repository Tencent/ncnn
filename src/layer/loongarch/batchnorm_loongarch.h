// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BATCHNORM_LOONGARCH_H
#define LAYER_BATCHNORM_LOONGARCH_H

#include "batchnorm.h"

namespace ncnn {

class BatchNorm_loongarch : public BatchNorm
{
public:
    BatchNorm_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BATCHNORM_LOONGARCH_H

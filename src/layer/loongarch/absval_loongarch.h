// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ABSVAL_LOONGARCH_H
#define LAYER_ABSVAL_LOONGARCH_H

#include "absval.h"

namespace ncnn {

class AbsVal_loongarch : public AbsVal
{
public:
    AbsVal_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ABSVAL_LOONGARCH_H

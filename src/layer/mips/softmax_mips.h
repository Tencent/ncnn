// Copyright 2020 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTMAX_MIPS_H
#define LAYER_SOFTMAX_MIPS_H

#include "softmax.h"

namespace ncnn {

class Softmax_mips : public Softmax
{
public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_MIPS_H

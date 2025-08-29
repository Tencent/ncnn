// Copyright 2019 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BIAS_MIPS_H
#define LAYER_BIAS_MIPS_H

#include "bias.h"

namespace ncnn {

class Bias_mips : public Bias
{
public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BIAS_MIPS_H

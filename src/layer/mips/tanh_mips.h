// Copyright 2020 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TANH_MIPS_H
#define LAYER_TANH_MIPS_H

#include "tanh.h"

namespace ncnn {

class TanH_mips : public TanH
{
public:
    TanH_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_TANH_MIPS_H

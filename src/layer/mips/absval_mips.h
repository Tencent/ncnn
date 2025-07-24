// Copyright 2019 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ABSVAL_MIPS_H
#define LAYER_ABSVAL_MIPS_H

#include "absval.h"

namespace ncnn {

class AbsVal_mips : public AbsVal
{
public:
    AbsVal_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ABSVAL_MIPS_H

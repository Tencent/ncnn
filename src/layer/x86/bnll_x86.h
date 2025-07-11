// Copyright 2022 JasonZhang892 <zqhy_0929@163.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BNLL_X86_H
#define LAYER_BNLL_X86_H

#include "bnll.h"

namespace ncnn {

class BNLL_x86 : public BNLL
{
public:
    BNLL_x86();
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
};

} // namespace ncnn

#endif // LAYER_BNLL_H

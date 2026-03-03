// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LRN_X86_H
#define LAYER_LRN_X86_H

#include "lrn.h"

namespace ncnn {

class LRN_x86 : public LRN
{
public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_LRN_X86_H

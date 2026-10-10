// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ABSVAL_H
#define LAYER_ABSVAL_H

#include "layer.h"

namespace ncnn {

class AbsVal : public Layer
{
public:
    AbsVal();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ABSVAL_H

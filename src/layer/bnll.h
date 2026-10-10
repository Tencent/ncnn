// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BNLL_H
#define LAYER_BNLL_H

#include "layer.h"

namespace ncnn {

class BNLL : public Layer
{
public:
    BNLL();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
};

} // namespace ncnn

#endif // LAYER_BNLL_H

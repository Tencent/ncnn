// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TANH_H
#define LAYER_TANH_H

#include "layer.h"

namespace ncnn {

class TanH : public Layer
{
public:
    TanH();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_TANH_H

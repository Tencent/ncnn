// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWISH_H
#define LAYER_SWISH_H

#include "layer.h"

namespace ncnn {

class Swish : public Layer
{
public:
    Swish();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SWISH_H

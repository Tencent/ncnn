// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MISH_H
#define LAYER_MISH_H

#include "layer.h"

namespace ncnn {

class Mish : public Layer
{
public:
    Mish();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MISH_H

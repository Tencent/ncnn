// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLATTEN_H
#define LAYER_FLATTEN_H

#include "layer.h"

namespace ncnn {

class Flatten : public Layer
{
public:
    Flatten();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_FLATTEN_H

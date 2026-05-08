// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHRINK_H
#define LAYER_SHRINK_H

#include "layer.h"

namespace ncnn {

class Shrink : public Layer
{
public:
    Shrink();

    virtual int load_param(const ParamDict& pd);
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float bias;
    float lambd;
};

} // namespace ncnn

#endif // LAYER_SHRINK_H

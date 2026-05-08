// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EXPANDDIMS_H
#define LAYER_EXPANDDIMS_H

#include "layer.h"

namespace ncnn {

class ExpandDims : public Layer
{
public:
    ExpandDims();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    Mat axes;
};

} // namespace ncnn

#endif // LAYER_EXPANDDIMS_H

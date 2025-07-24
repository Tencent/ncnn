// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SLICE_H
#define LAYER_SLICE_H

#include "layer.h"

namespace ncnn {

class Slice : public Layer
{
public:
    Slice();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    Mat slices;
    Mat indices;
    int axis;
};

} // namespace ncnn

#endif // LAYER_SLICE_H

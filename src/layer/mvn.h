// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MVN_H
#define LAYER_MVN_H

#include "layer.h"

namespace ncnn {

class MVN : public Layer
{
public:
    MVN();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int normalize_variance;
    int across_channels;
    float eps;
};

} // namespace ncnn

#endif // LAYER_MVN_H

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_THRESHOLD_H
#define LAYER_THRESHOLD_H

#include "layer.h"

namespace ncnn {

class Threshold : public Layer
{
public:
    Threshold();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float threshold;
};

} // namespace ncnn

#endif // LAYER_THRESHOLD_H

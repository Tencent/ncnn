// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SQUEEZE_H
#define LAYER_SQUEEZE_H

#include "layer.h"

namespace ncnn {

class Squeeze : public Layer
{
public:
    Squeeze();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int squeeze_w;
    int squeeze_h;
    int squeeze_d;
    int squeeze_c;
    Mat axes;
};

} // namespace ncnn

#endif // LAYER_SQUEEZE_H

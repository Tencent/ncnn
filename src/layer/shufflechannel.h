// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHUFFLECHANNEL_H
#define LAYER_SHUFFLECHANNEL_H

#include "layer.h"

namespace ncnn {

class ShuffleChannel : public Layer
{
public:
    ShuffleChannel();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int group;
    int reverse;
};

} // namespace ncnn

#endif // LAYER_SHUFFLECHANNEL_H

// Copyright 2016 SoundAI Technology Co., Ltd. (author: Charles Wang)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_STATISTICSPOOLING_H
#define LAYER_STATISTICSPOOLING_H

#include "layer.h"

namespace ncnn {

class StatisticsPooling : public Layer
{
public:
    StatisticsPooling();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // param
    int include_stddev;
};

} // namespace ncnn

#endif // LAYER_STATISTICSPOOLING_H

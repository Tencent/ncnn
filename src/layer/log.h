// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LOG_H
#define LAYER_LOG_H

#include "layer.h"

namespace ncnn {

class Log : public Layer
{
public:
    Log();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float base;
    float scale;
    float shift;
};

} // namespace ncnn

#endif // LAYER_LOG_H

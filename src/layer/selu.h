// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SELU_H
#define LAYER_SELU_H

#include "layer.h"

namespace ncnn {

class SELU : public Layer
{
public:
    SELU();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float alpha;
    float lambda;
};

} // namespace ncnn

#endif // LAYER_SELU_H

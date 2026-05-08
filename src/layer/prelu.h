// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PRELU_H
#define LAYER_PRELU_H

#include "layer.h"

namespace ncnn {

class PReLU : public Layer
{
public:
    PReLU();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    int num_slope;
    Mat slope_data;
};

} // namespace ncnn

#endif // LAYER_PRELU_H

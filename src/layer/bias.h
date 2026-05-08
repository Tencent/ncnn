// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BIAS_H
#define LAYER_BIAS_H

#include "layer.h"

namespace ncnn {

class Bias : public Layer
{
public:
    Bias();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    // param
    int bias_data_size;

    // model
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_BIAS_H

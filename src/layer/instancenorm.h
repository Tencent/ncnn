// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INSTANCENORM_H
#define LAYER_INSTANCENORM_H

#include "layer.h"

namespace ncnn {

class InstanceNorm : public Layer
{
public:
    InstanceNorm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    // param
    int channels;
    float eps;
    int affine;

    // model
    Mat gamma_data;
    Mat beta_data;
};

} // namespace ncnn

#endif // LAYER_INSTANCENORM_H

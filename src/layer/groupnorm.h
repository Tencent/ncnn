// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GROUPNORM_H
#define LAYER_GROUPNORM_H

#include "layer.h"

namespace ncnn {

class GroupNorm : public Layer
{
public:
    GroupNorm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    // param
    int group;
    int channels;
    float eps;
    int affine;

    // model
    Mat gamma_data;
    Mat beta_data;
};

} // namespace ncnn

#endif // LAYER_GROUPNORM_H

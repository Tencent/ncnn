// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BATCHNORM_H
#define LAYER_BATCHNORM_H

#include "layer.h"

namespace ncnn {

class BatchNorm : public Layer
{
public:
    BatchNorm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    // param
    int channels;
    float eps;

    // model
    Mat slope_data;
    Mat mean_data;
    Mat var_data;
    Mat bias_data;

    Mat a_data;
    Mat b_data;
};

} // namespace ncnn

#endif // LAYER_BATCHNORM_H

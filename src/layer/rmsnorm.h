// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RMSNORM_H
#define LAYER_RMSNORM_H

#include "layer.h"

namespace ncnn {

class RMSNorm : public Layer
{
public:
    RMSNorm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    int affine_size;
    float eps;
    int affine;

    Mat gamma_data;
};

} // namespace ncnn

#endif // LAYER_RMSNORM_H

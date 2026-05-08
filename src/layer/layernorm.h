// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LAYERNORM_H
#define LAYER_LAYERNORM_H

#include "layer.h"

namespace ncnn {

class LayerNorm : public Layer
{
public:
    LayerNorm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    // param
    int affine_size;
    float eps;
    int affine;

    // model
    Mat gamma_data;
    Mat beta_data;
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_H

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EMBED_H
#define LAYER_EMBED_H

#include "layer.h"

namespace ncnn {

class Embed : public Layer
{
public:
    Embed();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // param
    int num_output;
    int input_dim;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // model
    Mat weight_data;
    Mat bias_data;

#if NCNN_INT8
    float weight_data_int8_scale;
#endif
};

} // namespace ncnn

#endif // LAYER_EMBED_H

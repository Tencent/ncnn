// Copyright 2019 BUG1989
// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REQUANTIZE_H
#define LAYER_REQUANTIZE_H

#include "layer.h"

namespace ncnn {

class Requantize : public Layer
{
public:
    Requantize();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int scale_in_data_size;
    int scale_out_data_size;
    int bias_data_size;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    Mat scale_in_data;
    Mat scale_out_data;
    Mat bias_data;

    //     float scale_in;  // bottom_blob_scale * weight_scale
    //     float scale_out; // top_blob_scale / (bottom_blob_scale * weight_scale)
    //     int bias_term;
    //     int bias_data_size;
    //
    //     Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_REQUANTIZE_H

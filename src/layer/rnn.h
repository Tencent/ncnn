// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RNN_H
#define LAYER_RNN_H

#include "layer.h"

namespace ncnn {

class RNN : public Layer
{
public:
    RNN();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int num_output;
    int weight_data_size;
    int direction; // 0=forward 1=reverse 2=bidirectional

    int int8_scale_term;

    Mat weight_hc_data;
    Mat weight_xc_data;
    Mat bias_c_data;

#if NCNN_INT8
    Mat weight_hc_data_int8_scales;
    Mat weight_xc_data_int8_scales;
#endif
};

} // namespace ncnn

#endif // LAYER_RNN_H

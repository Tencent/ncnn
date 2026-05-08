// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTIONDEPTHWISE3D_H
#define LAYER_DECONVOLUTIONDEPTHWISE3D_H

#include "layer.h"

namespace ncnn {

class DeconvolutionDepthWise3D : public Layer
{
public:
    DeconvolutionDepthWise3D();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    void cut_padding(const Mat& top_blob_bordered, Mat& top_blob, const Option& opt) const;

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int kernel_d;
    int dilation_w;
    int dilation_h;
    int dilation_d;
    int stride_w;
    int stride_h;
    int stride_d;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int pad_front;
    int pad_behind;
    int output_pad_right;
    int output_pad_bottom;
    int output_pad_behind;
    int output_w;
    int output_h;
    int output_d;
    int bias_term;

    int weight_data_size;
    int group;

    int activation_type;
    Mat activation_params;

    // model
    Mat weight_data;
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTIONDEPTHWISE3D_H

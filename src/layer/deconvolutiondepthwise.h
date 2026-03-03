// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTIONDEPTHWISE_H
#define LAYER_DECONVOLUTIONDEPTHWISE_H

#include "layer.h"

namespace ncnn {

class DeconvolutionDepthWise : public Layer
{
public:
    DeconvolutionDepthWise();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void cut_padding(const Mat& top_blob_bordered, Mat& top_blob, const Option& opt) const;

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int output_pad_right;
    int output_pad_bottom;
    int output_w;
    int output_h;
    int bias_term;

    int weight_data_size;
    int group;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTIONDEPTHWISE_H

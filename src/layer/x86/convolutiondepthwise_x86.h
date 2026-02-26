// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTIONDEPTHWISE_X86_H
#define LAYER_CONVOLUTIONDEPTHWISE_X86_H

#include "convolutiondepthwise.h"

namespace ncnn {

class ConvolutionDepthWise_x86 : public ConvolutionDepthWise
{
public:
    ConvolutionDepthWise_x86();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int create_group_ops(const Option& opt);
#if NCNN_INT8
    int create_pipeline_int8_x86(const Option& opt);
    int forward_int8_x86(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    Layer* activation;
    std::vector<ncnn::Layer*> group_ops;

    Mat weight_data_tm;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_X86_H

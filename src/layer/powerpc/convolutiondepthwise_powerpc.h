// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTIONDEPTHWISE_POWERPC_H
#define LAYER_CONVOLUTIONDEPTHWISE_POWERPC_H

#include "convolutiondepthwise.h"

namespace ncnn {

class ConvolutionDepthWise_powerpc : public ConvolutionDepthWise
{
public:
    ConvolutionDepthWise_powerpc();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    // the generic kernel is not packing aware
    int forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    Layer* activation;

    // pack4 depthwise weights, each kernel tap is one vector of 4 channels
    Mat weight_data_tm;
    int use_packed_weights;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_POWERPC_H

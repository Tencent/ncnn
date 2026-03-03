// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTION_ARM_H
#define LAYER_DECONVOLUTION_ARM_H

#include "deconvolution.h"

namespace ncnn {

class Deconvolution_arm : public Deconvolution
{
public:
    Deconvolution_arm();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ARM82
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int create_pipeline_bf16s(const Option& opt);
    int forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    Layer* activation;
    Layer* gemm;

    Mat weight_data_tm;

    // fp16
    Mat bias_data_fp16;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTION_ARM_H

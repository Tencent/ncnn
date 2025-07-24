// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTIONDEPTHWISE_LOONGARCH_H
#define LAYER_DECONVOLUTIONDEPTHWISE_LOONGARCH_H

#include "deconvolutiondepthwise.h"

namespace ncnn {

class DeconvolutionDepthWise_loongarch : public DeconvolutionDepthWise
{
public:
    DeconvolutionDepthWise_loongarch();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int create_group_ops(const Option& opt);

public:
    std::vector<ncnn::Layer*> group_ops;

    Mat weight_data_tm;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTIONDEPTHWISE_LOONGARCH_H

// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION_LOONGARCH_H
#define LAYER_CONVOLUTION_LOONGARCH_H

#include "convolution.h"

namespace ncnn {

class Convolution_loongarch : public Convolution
{
public:
    Convolution_loongarch();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_INT8
    int create_pipeline_int8_loongarch(const Option& opt);
    int forward_int8_loongarch(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    Layer* activation;

    Mat weight_data_tm;
    Mat weight_sgemm_data;
    Mat weight_winograd23_data;
    Mat weight_winograd43_data;
    Mat weight_winograd63_data;

#if NCNN_INT8
    Mat scale_in_data;
#endif
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_LOONGARCH_H

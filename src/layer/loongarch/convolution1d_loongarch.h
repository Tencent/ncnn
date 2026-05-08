// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION1D_LOONGARCH_H
#define LAYER_CONVOLUTION1D_LOONGARCH_H

#include "convolution1d.h"

namespace ncnn {

class Convolution1D_loongarch : public Convolution1D
{
public:
    Convolution1D_loongarch();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // packn
    Mat weight_data_packed;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION1D_LOONGARCH_H

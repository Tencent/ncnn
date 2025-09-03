// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTION_MIPS_H
#define LAYER_DECONVOLUTION_MIPS_H

#include "deconvolution.h"

namespace ncnn {

class Deconvolution_mips : public Deconvolution
{
public:
    Deconvolution_mips();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    Mat weight_data_tm;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTION_MIPS_H

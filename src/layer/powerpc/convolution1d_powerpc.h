// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION1D_POWERPC_H
#define LAYER_CONVOLUTION1D_POWERPC_H

#include "convolution1d.h"

namespace ncnn {

class Convolution1D_powerpc : public Convolution1D
{
public:
    Convolution1D_powerpc();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    // the generic kernel is not packing aware
    int forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // pack4 weights, layout [outpack][inpack][kernel_w][4 out lane][4 in lane]
    Mat weight_data_tm;
    int use_packed_gemm;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION1D_POWERPC_H

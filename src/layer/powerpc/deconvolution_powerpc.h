// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTION_POWERPC_H
#define LAYER_DECONVOLUTION_POWERPC_H

#include "deconvolution.h"

namespace ncnn {

class Deconvolution_powerpc : public Deconvolution
{
public:
    Deconvolution_powerpc();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    // the generic kernel is not packing aware
    int forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // pack4 weights with spatially flipped taps
    // layout [maxk][num_input/4][num_output/4][4 in lane][4 out lane]
    Mat weight_data_tm;
    int use_packed_gemm;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTION_POWERPC_H

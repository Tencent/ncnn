// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTIONDEPTHWISE1D_POWERPC_H
#define LAYER_CONVOLUTIONDEPTHWISE1D_POWERPC_H

#include "convolutiondepthwise1d.h"

namespace ncnn {

class ConvolutionDepthWise1D_powerpc : public ConvolutionDepthWise1D
{
public:
    ConvolutionDepthWise1D_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    // the generic kernel is not packing aware
    int forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE1D_POWERPC_H

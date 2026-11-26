// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTIONDEPTHWISE3D_POWERPC_H
#define LAYER_DECONVOLUTIONDEPTHWISE3D_POWERPC_H

#include "deconvolutiondepthwise3d.h"

namespace ncnn {

class DeconvolutionDepthWise3D_powerpc : public DeconvolutionDepthWise3D
{
public:
    DeconvolutionDepthWise3D_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    // the generic kernel is not packing aware
    int forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTIONDEPTHWISE3D_POWERPC_H

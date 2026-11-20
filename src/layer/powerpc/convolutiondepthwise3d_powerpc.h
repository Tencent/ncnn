// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTIONDEPTHWISE3D_POWERPC_H
#define LAYER_CONVOLUTIONDEPTHWISE3D_POWERPC_H

#include "convolutiondepthwise3d.h"

namespace ncnn {

class ConvolutionDepthWise3D_powerpc : public ConvolutionDepthWise3D
{
public:
    ConvolutionDepthWise3D_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    // the generic kernel is not packing aware
    int forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE3D_POWERPC_H

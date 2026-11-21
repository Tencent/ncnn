// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION3D_POWERPC_H
#define LAYER_CONVOLUTION3D_POWERPC_H

#include "convolution3d.h"

namespace ncnn {

class Convolution3D_powerpc : public Convolution3D
{
public:
    Convolution3D_powerpc();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    // the generic kernel is not packing aware
    int forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
public:
    // pack4 weights, layout [maxk][inpack][outpack][4 in lane][4 out lane]
    Mat weight_data_tm;
    int use_packed_gemm;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION3D_POWERPC_H

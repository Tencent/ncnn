// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_COPYTO_H
#define LAYER_COPYTO_H

#include "layer.h"

namespace ncnn {

class CopyTo : public Layer
{
public:
    CopyTo();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void resolve_copyto_offset(const Mat& self_blob, int& woffset, int& hoffset, int& doffset, int& coffset) const;

public:
    int woffset;
    int hoffset;
    int doffset;
    int coffset;

    // numpy-style slice
    // if provided, all the above attributes will be ignored
    Mat starts;
    Mat axes;
};

} // namespace ncnn

#endif // LAYER_COPYTO_H

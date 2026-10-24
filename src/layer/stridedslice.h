// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_STRIDEDSLICE_H
#define LAYER_STRIDEDSLICE_H

#include "layer.h"

namespace ncnn {

class StridedSlice : public Layer
{
public:
    StridedSlice();

    virtual int load_param(const ParamDict& pd);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int forward_slice(const Mat& bottom_blob, Mat& top_blob, const int* starts, const int* ends, const Option& opt) const;

public:
    int begin_mask;
    int end_mask;
    Mat starts;
    Mat ends;
    Mat axes;
    Mat steps;

    // see docs/developer-guide/expression.md
    std::string starts_expr;
    std::string ends_expr;
};

} // namespace ncnn

#endif // LAYER_STRIDEDSLICE_H

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INTERP_H
#define LAYER_INTERP_H

#include "layer.h"

namespace ncnn {

class Interp : public Layer
{
public:
    Interp();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int eval_size_expr(const std::vector<Mat>& bottom_blobs, int& outw, int& outh) const;

public:
    // param
    int resize_type; //1=nearest  2=bilinear  3=bicubic
    float width_scale;
    float height_scale;
    int output_width;
    int output_height;
    int dynamic_target_size;
    int align_corner;

    // see docs/developer-guide/expression.md
    std::string size_expr;
};

} // namespace ncnn

#endif // LAYER_INTERP_H

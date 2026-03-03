// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CROP_H
#define LAYER_CROP_H

#include "layer.h"

namespace ncnn {

class Crop : public Layer
{
public:
    Crop();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void resolve_crop_roi(const Mat& bottom_blob, int& woffset, int& hoffset, int& doffset, int& coffset, int& outw, int& outh, int& outd, int& outc) const;
    void resolve_crop_roi(const Mat& bottom_blob, const Mat& reference_blob, int& woffset, int& hoffset, int& doffset, int& coffset, int& outw, int& outh, int& outd, int& outc) const;
    void resolve_crop_roi(const Mat& bottom_blob, const int* param_data, int& woffset, int& hoffset, int& doffset, int& coffset, int& outw, int& outh, int& outd, int& outc) const;
    int eval_crop_expr(const std::vector<Mat>& bottom_blobs, int& woffset, int& hoffset, int& doffset, int& coffset, int& outw, int& outh, int& outd, int& outc) const;

public:
    // -233 = dynamic offset from reference blob
    int woffset;
    int hoffset;
    int doffset;
    int coffset;

    // -233 = remaining
    int outw;
    int outh;
    int outd;
    int outc;

    // tail offset for cropping, ignored if reference blob is provided
    // woffset is aka left, and woffset2 is aka right
    int woffset2;
    int hoffset2;
    int doffset2;
    int coffset2;

    // numpy-style slice
    // if provided, all the above attributes will be ignored
    Mat starts;
    Mat ends;
    Mat axes;

    // see docs/developer-guide/expression.md
    std::string starts_expr;
    std::string ends_expr;
    std::string axes_expr;
};

} // namespace ncnn

#endif // LAYER_CROP_H

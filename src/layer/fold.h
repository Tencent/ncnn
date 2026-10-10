// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FOLD_H
#define LAYER_FOLD_H

#include "layer.h"

namespace ncnn {

class Fold : public Layer
{
public:
    Fold();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left; // -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    int pad_top;
    int pad_bottom;
    int output_w;
    int output_h;
};

} // namespace ncnn

#endif // LAYER_FOLD_H

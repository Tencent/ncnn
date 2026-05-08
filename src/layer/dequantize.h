// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEQUANTIZE_H
#define LAYER_DEQUANTIZE_H

#include "layer.h"

namespace ncnn {

class Dequantize : public Layer
{
public:
    Dequantize();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int scale_data_size;
    int bias_data_size;

    Mat scale_data;
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_DEQUANTIZE_H

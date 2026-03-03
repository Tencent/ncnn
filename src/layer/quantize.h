// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_QUANTIZE_H
#define LAYER_QUANTIZE_H

#include "layer.h"

namespace ncnn {

class Quantize : public Layer
{
public:
    Quantize();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int scale_data_size;
    Mat scale_data;
};

} // namespace ncnn

#endif // LAYER_QUANTIZE_H

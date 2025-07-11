// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PADDING_H
#define LAYER_PADDING_H

#include "layer.h"

namespace ncnn {

class Padding : public Layer
{
public:
    Padding();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int top;
    int bottom;
    int left;
    int right;
    int type; // 0=CONSTANT 1=REPLICATE 2=REFLECT
    float value;
    int front;
    int behind;

    // per channel pad value
    int per_channel_pad_data_size;
    Mat per_channel_pad_data;
};

} // namespace ncnn

#endif // LAYER_PADDING_H

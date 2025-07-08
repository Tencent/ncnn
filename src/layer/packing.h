// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PACKING_H
#define LAYER_PACKING_H

#include "layer.h"

namespace ncnn {

class Packing : public Layer
{
public:
    Packing();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int out_elempack;
    int use_padding;

    // element type
    // 0 = auto
    // 1 = fp32
    // 2 = fp16
    // 3 = int32
    // 4 = int8
    int cast_type_from;
    int cast_type_to;
};

} // namespace ncnn

#endif // LAYER_PACKING_H

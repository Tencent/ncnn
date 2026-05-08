// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DROPOUT_H
#define LAYER_DROPOUT_H

#include "layer.h"

namespace ncnn {

class Dropout : public Layer
{
public:
    Dropout();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float scale;
};

} // namespace ncnn

#endif // LAYER_DROPOUT_H

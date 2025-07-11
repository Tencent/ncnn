// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GELU_H
#define LAYER_GELU_H

#include "layer.h"

namespace ncnn {

class GELU : public Layer
{
public:
    GELU();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    int fast_gelu;
};

} // namespace ncnn

#endif // LAYER_GELU_H

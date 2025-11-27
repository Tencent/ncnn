// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CELU_H
#define LAYER_CELU_H

#include "layer.h"

namespace ncnn {

class CELU : public Layer
{
public:
    CELU();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float alpha;
};

} // namespace ncnn

#endif // LAYER_CELU_H

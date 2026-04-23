// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_WHERE_H
#define LAYER_WHERE_H

#include "layer.h"

namespace ncnn {

class Where : public Layer
{
public:
    Where();

    virtual int load_param(const ParamDict& pd);

    using Layer::forward;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int with_scalar_a;
    float a;
    int with_scalar_b;
    float b;
};

} // namespace ncnn

#endif // LAYER_WHERE_H

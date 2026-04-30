// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EXPAND_H
#define LAYER_EXPAND_H

#include "layer.h"

namespace ncnn {

class Expand : public Layer
{
public:
    Expand();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_EXPAND_H

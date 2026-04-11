// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GATHERELEMENTS_H
#define LAYER_GATHERELEMENTS_H

#include "layer.h"

namespace ncnn {

class GatherElements : public Layer
{
public:
    GatherElements();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // param_0 = axis (default 0)
    int axis;
};

} // namespace ncnn

#endif // LAYER_GATHERELEMENTS_H

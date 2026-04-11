// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TOPK_H
#define LAYER_TOPK_H

#include "layer.h"

namespace ncnn {

class TopK : public Layer
{
public:
    TopK();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int axis;
    int largest;
    int sorted;
    int k;
};

} // namespace ncnn

#endif // LAYER_TOPK_H

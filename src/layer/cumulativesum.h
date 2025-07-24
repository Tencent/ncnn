// Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CUMULATIVESUM_H
#define LAYER_CUMULATIVESUM_H

#include "layer.h"

namespace ncnn {

class CumulativeSum : public Layer
{
public:
    CumulativeSum();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    int axis;
};

} // namespace ncnn

#endif // LAYER_CUMULATIVESUM_H

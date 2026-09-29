// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONCAT_H
#define LAYER_CONCAT_H

#include "layer.h"

namespace ncnn {

class Concat : public Layer
{
public:
    Concat();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    // Reject inputs whose non-concat dimensions disagree. Without this guard,
    // axis-0 paths size the output from bottom_blobs[0] then memcpy each
    // input's total() and can heap-overflow (see #7025).
    int check_shape(const std::vector<Mat>& bottom_blobs) const;

public:
    int axis;
};

} // namespace ncnn

#endif // LAYER_CONCAT_H

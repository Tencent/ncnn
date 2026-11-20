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

#if NCNN_BATCH
protected:
    static bool same_shape_except_axis(const Mat& a, const Mat& b, int axis);
    int forward_batch(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int axis;
};

} // namespace ncnn

#endif // LAYER_CONCAT_H

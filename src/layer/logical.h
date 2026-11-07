// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LOGICAL_H
#define LAYER_LOGICAL_H

#include "layer.h"

namespace ncnn {

class Logical : public Layer
{
public:
    Logical();

    virtual int load_param(const ParamDict& pd);

    using Layer::forward;
    using Layer::forward_inplace;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    enum OperationType
    {
        Operation_NOT = 0,
        Operation_AND = 1,
        Operation_OR = 2,
        Operation_XOR = 3
    };

public:
    int op_type;
    int with_scalar;
    signed char b;
};

} // namespace ncnn

#endif // LAYER_LOGICAL_H

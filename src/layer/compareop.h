// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_COMPAREOP_H
#define LAYER_COMPAREOP_H

#include "layer.h"

namespace ncnn {

class CompareOp : public Layer
{
public:
    CompareOp();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    using Layer::forward;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    enum OperationType
    {
        Operation_LT = 0, // <
        Operation_GT = 1, // >
        Operation_LE = 2, // <=
        Operation_GE = 3, // >=
        Operation_EQ = 4, // ==
        Operation_NE = 5  // !=
    };

public:
    // param
    int op_type;
    int with_scalar;
    float b;
};

} // namespace ncnn

#endif // LAYER_COMPAREOP_H
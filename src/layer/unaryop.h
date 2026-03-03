// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNARYOP_H
#define LAYER_UNARYOP_H

#include "layer.h"

namespace ncnn {

class UnaryOp : public Layer
{
public:
    UnaryOp();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    enum OperationType
    {
        Operation_ABS = 0,
        Operation_NEG = 1,
        Operation_FLOOR = 2,
        Operation_CEIL = 3,
        Operation_SQUARE = 4,
        Operation_SQRT = 5,
        Operation_RSQRT = 6,
        Operation_EXP = 7,
        Operation_LOG = 8,
        Operation_SIN = 9,
        Operation_COS = 10,
        Operation_TAN = 11,
        Operation_ASIN = 12,
        Operation_ACOS = 13,
        Operation_ATAN = 14,
        Operation_RECIPROCAL = 15,
        Operation_TANH = 16,
        Operation_LOG10 = 17,
        Operation_ROUND = 18,
        Operation_TRUNC = 19
    };

public:
    // param
    int op_type;
};

} // namespace ncnn

#endif // LAYER_UNARYOP_H

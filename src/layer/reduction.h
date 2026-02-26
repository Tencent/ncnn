// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REDUCTION_H
#define LAYER_REDUCTION_H

#include "layer.h"

namespace ncnn {

class Reduction : public Layer
{
public:
    Reduction();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    enum ReductionOp
    {
        ReductionOp_SUM = 0,
        ReductionOp_ASUM = 1,
        ReductionOp_SUMSQ = 2,
        ReductionOp_MEAN = 3,
        ReductionOp_MAX = 4,
        ReductionOp_MIN = 5,
        ReductionOp_PROD = 6,
        ReductionOp_L1 = 7,
        ReductionOp_L2 = 8,
        ReductionOp_LogSum = 9,
        ReductionOp_LogSumExp = 10
    };

public:
    // param
    int operation;
    int reduce_all;
    float coeff;
    Mat axes;
    int keepdims;
};

} // namespace ncnn

#endif // LAYER_REDUCTION_H

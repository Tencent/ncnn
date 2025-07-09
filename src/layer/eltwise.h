// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELTWISE_H
#define LAYER_ELTWISE_H

#include "layer.h"

namespace ncnn {

class Eltwise : public Layer
{
public:
    Eltwise();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    enum OperationType
    {
        Operation_PROD = 0,
        Operation_SUM = 1,
        Operation_MAX = 2
    };

public:
    // param
    int op_type;
    Mat coeffs;
};

} // namespace ncnn

#endif // LAYER_ELTWISE_H

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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

    enum {
        ReductionOp_SUM     = 0,
        ReductionOp_ASUM    = 1,
        ReductionOp_SUMSQ   = 2,
        ReductionOp_MEAN    = 3,
        ReductionOp_MAX     = 4,
        ReductionOp_MIN     = 5,
        ReductionOp_PROD    = 6
    };

public:
    // param
    int operation;
    int dim;
    float coeff;
};

} // namespace ncnn

#endif // LAYER_REDUCTION_H

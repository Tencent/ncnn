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

    enum {
        Operation_ABS   = 0,
        Operation_NEG   = 1,
        Operation_FLOOR = 2,
        Operation_CEIL  = 3,
        Operation_SQUARE= 4,
        Operation_SQRT  = 5,
        Operation_RSQRT = 6,
        Operation_EXP   = 7,
        Operation_LOG   = 8,
        Operation_SIN   = 9,
        Operation_COS   = 10,
        Operation_TAN   = 11,
        Operation_ASIN  = 12,
        Operation_ACOS  = 13,
        Operation_ATAN  = 14,
        Operation_RECIPROCAL = 15
    };

public:
    // param
    int op_type;
};

} // namespace ncnn

#endif // LAYER_UNARYOP_H

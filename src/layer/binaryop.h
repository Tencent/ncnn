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

#ifndef LAYER_BINARYOP_H
#define LAYER_BINARYOP_H

#include "layer.h"

namespace ncnn {

class BinaryOp : public Layer
{
public:
    BinaryOp();

    virtual int load_param(const ParamDict& pd);

    using Layer::forward;
    using Layer::forward_inplace;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    enum OperationType
    {
        Operation_ADD = 0,
        Operation_SUB = 1,
        Operation_MUL = 2,
        Operation_DIV = 3,
        Operation_MAX = 4,
        Operation_MIN = 5,
        Operation_POW = 6,
        Operation_RSUB = 7,
        Operation_RDIV = 8
    };

public:
    // param
    int op_type;
    int with_scalar;
    float b;
};

} // namespace ncnn

#endif // LAYER_BINARYOP_H

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

#ifndef LAYER_CONVOLUTION_ARM_H
#define LAYER_CONVOLUTION_ARM_H

#include "convolution.h"

namespace ncnn {

class Convolution_arm : public Convolution
{
public:
    virtual int load_param(const ParamDict& pd);

#if NCNN_STDIO
    virtual int load_model(FILE* binfp);
#endif // NCNN_STDIO
    virtual int load_model(const unsigned char*& mem);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob) const;

public:
    bool use_winograd3x3;
    Mat weight_3x3_winograd64_data;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_ARM_H

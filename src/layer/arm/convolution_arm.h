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

typedef void (*conv_func)(const Mat&, Mat&, const Mat&, const Mat&, const Option&);

class Convolution_arm : public Convolution
{
public:
    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forwardDilation(const Mat& bottom_blob, Mat& top_blob, conv_func conv, const Option& opt) const;

public:
    bool use_winograd3x3;
    bool use_sgemm1x1;
    Mat weight_3x3_winograd64_data;
    Mat weight_1x1_sgemm_data;
    Mat weight_3x3s2_data;
    Mat weight_3x3s1_int8_data;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_ARM_H

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_GEMM_H
#define LAYER_GEMM_H

#include "layer.h"

namespace ncnn {

class Gemm : public Layer
{
public:
    Gemm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    float alpha;
    float beta;
    int transA;
    int transB;

    int constantA;
    int constantB;
    int constantC;
    int constantM;
    int constantN;
    int constantK;
    int constant_broadcast_type_C;
    int output_N1M;
    int output_elempack;
    int output_elemtype; // 0=auto 1=fp32
    int output_transpose;

    int constant_TILE_M;
    int constant_TILE_N;
    int constant_TILE_K;

    // constant A / B / C
    Mat A_data;
    Mat B_data;
    Mat C_data;
};

} // namespace ncnn

#endif // LAYER_GEMM_H

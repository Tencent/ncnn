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

#ifndef LAYER_RNN_H
#define LAYER_RNN_H

#include "layer.h"

namespace ncnn {

class RNN : public Layer
{
public:
    RNN();
    virtual ~RNN();

#if NCNN_STDIO
#if NCNN_STRING
    virtual int load_param(FILE* paramfp);
#endif // NCNN_STRING
    virtual int load_param_bin(FILE* paramfp);
    virtual int load_model(FILE* binfp);
#endif // NCNN_STDIO
    virtual int load_param(const unsigned char*& mem);
    virtual int load_model(const unsigned char*& mem);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

public:
    // param
    int num_output;
    int weight_data_size;

    // model
    Mat weight_hh_data;
    Mat weight_xh_data;
    Mat weight_ho_data;
    Mat bias_h_data;
    Mat bias_o_data;
};

} // namespace ncnn

#endif // LAYER_RNN_H

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

#ifndef LAYER_SCALE_H
#define LAYER_SCALE_H

#include "layer.h"

namespace ncnn {

class Scale : public Layer
{
public:
    Scale();
    virtual ~Scale();

#if NCNN_STDIO
#if NCNN_STRING
    virtual int load_param(FILE* paramfp);
#endif // NCNN_STRING
    virtual int load_param_bin(FILE* paramfp);
    virtual int load_model(FILE* binfp);
#endif // NCNN_STDIO
    virtual int load_param(const unsigned char*& mem);
    virtual int load_model(const unsigned char*& mem);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob) const;

    virtual int forward_inplace(Mat& bottom_top_blob) const;

public:
    // param
    int scale_data_size;
    int bias_term;

    // model
    Mat scale_data;
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_SCALE_H

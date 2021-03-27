// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_REQUANTIZE_H
#define LAYER_REQUANTIZE_H

#include "layer.h"

namespace ncnn {

class Requantize : public Layer
{
public:
    Requantize();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int scale_in_data_size;
    int scale_out_data_size;
    int bias_data_size;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    Mat scale_in_data;
    Mat scale_out_data;
    Mat bias_data;

    //     float scale_in;  // bottom_blob_scale * weight_scale
    //     float scale_out; // top_blob_scale / (bottom_blob_scale * weight_scale)
    //     int bias_term;
    //     int bias_data_size;
    //
    //     Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_REQUANTIZE_H
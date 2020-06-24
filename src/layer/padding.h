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

#ifndef LAYER_PADDING_H
#define LAYER_PADDING_H

#include "layer.h"

namespace ncnn {

class Padding : public Layer
{
public:
    Padding();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // -233 = dynamic offset from reference blob
    int top;
    int bottom;
    int left;
    int right;
    int type; // 0=CONSTANT 1=REPLICATE 2=REFLECT
    float value;
    int front;
    int behind;

    // per channel pad value
    int per_channel_pad_data_size;
    Mat per_channel_pad_data;
};

} // namespace ncnn

#endif // LAYER_PADDING_H

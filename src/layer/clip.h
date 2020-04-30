// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_CLIP_H
#define LAYER_CLIP_H

#include "layer.h"

namespace ncnn {

class Clip : public Layer
{
public:
    Clip();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
    
    virtual int forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const;

public:
    float min;
    float max;
};

} // namespace ncnn

#endif // LAYER_CLIP_H

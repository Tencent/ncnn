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

#ifndef LAYER_SHUFFLECHANNEL_H
#define LAYER_SHUFFLECHANNEL_H

#include "layer.h"

namespace ncnn {

class ShuffleChannel : public Layer
{
public:
    ShuffleChannel();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

#if NCNN_VULKAN
    virtual int create_pipeline();
    virtual int destroy_pipeline();

    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

public:
    int group;

#if NCNN_VULKAN
    Pipeline* pipeline_shufflechannel;
    Pipeline* pipeline_shufflechannel_pack4;
#endif // NCNN_VULKAN
};

} // namespace ncnn

#endif // LAYER_SHUFFLECHANNEL_H

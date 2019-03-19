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

#ifndef LAYER_LRN_H
#define LAYER_LRN_H

#include "layer.h"

namespace ncnn {

class LRN : public Layer
{
public:
    LRN();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_VULKAN
    virtual int create_pipeline();
    virtual int destroy_pipeline();

    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

    enum { NormRegion_ACROSS_CHANNELS = 0, NormRegion_WITHIN_CHANNEL = 1 };

public:
    // param
    int region_type;
    int local_size;
    float alpha;
    float beta;
    float bias;

#if NCNN_VULKAN
    Pipeline* pipeline_lrn_square_pad;
    Pipeline* pipeline_lrn_norm;
    Pipeline* pipeline_lrn_square_pad_across_channel_pack4;
    Pipeline* pipeline_lrn_norm_across_channel_pack4;
    Pipeline* pipeline_lrn_square_pad_within_channel_pack4;
    Pipeline* pipeline_lrn_norm_within_channel_pack4;
#endif // NCNN_VULKAN
};

} // namespace ncnn

#endif // LAYER_LRN_H

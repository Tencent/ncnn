// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_CONVOLUTIONDEPTHWISE_VULKAN_H
#define LAYER_CONVOLUTIONDEPTHWISE_VULKAN_H

#include "convolutiondepthwise.h"

namespace ncnn {

class ConvolutionDepthWise_vulkan : public ConvolutionDepthWise
{
public:
    ConvolutionDepthWise_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using ConvolutionDepthWise::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Mat weight_data_packed;
    Mat weight_data_packed_groups;
    Mat bias_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    VkImageMat weight_data_gpu_image;
    VkImageMat bias_data_gpu_image;

    ncnn::Layer* padding;

    Pipeline* pipeline_convolutiondepthwise;
    Pipeline* pipeline_convolutiondepthwise_pack4;
    Pipeline* pipeline_convolutiondepthwise_pack8;

    Pipeline* pipeline_convolutiondepthwise_group;
    Pipeline* pipeline_convolutiondepthwise_group_pack4;
    Pipeline* pipeline_convolutiondepthwise_group_pack1to4;
    Pipeline* pipeline_convolutiondepthwise_group_pack4to1;
    Pipeline* pipeline_convolutiondepthwise_group_pack8;
    Pipeline* pipeline_convolutiondepthwise_group_pack1to8;
    Pipeline* pipeline_convolutiondepthwise_group_pack4to8;
    Pipeline* pipeline_convolutiondepthwise_group_pack8to4;
    Pipeline* pipeline_convolutiondepthwise_group_pack8to1;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_VULKAN_H

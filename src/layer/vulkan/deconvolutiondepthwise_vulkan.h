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

#ifndef LAYER_DECONVOLUTIONDEPTHWISE_VULKAN_H
#define LAYER_DECONVOLUTIONDEPTHWISE_VULKAN_H

#include "deconvolutiondepthwise.h"

namespace ncnn {

class DeconvolutionDepthWise_vulkan : virtual public DeconvolutionDepthWise
{
public:
    DeconvolutionDepthWise_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using DeconvolutionDepthWise::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    ncnn::Layer* crop;
    ncnn::Layer* output_pad;
    ncnn::Layer* output_crop;
    ncnn::Layer* packing_pack1;
    ncnn::Layer* packing_pack4;

    Pipeline* pipeline_deconvolutiondepthwise;

    // pack4
    VkMat weight_data_gpu_pack4;
    VkMat bias_data_gpu_pack4;

    Pipeline* pipeline_deconvolutiondepthwise_pack4;

    Pipeline* pipeline_deconvolutiondepthwise_group;
    Pipeline* pipeline_deconvolutiondepthwise_group_pack4;

    // pack1to4
    VkMat weight_data_gpu_pack1to4;
    Pipeline* pipeline_deconvolutiondepthwise_group_pack1to4;

    // pack4to1
    VkMat weight_data_gpu_pack4to1;
    Pipeline* pipeline_deconvolutiondepthwise_group_pack4to1;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTIONDEPTHWISE_VULKAN_H

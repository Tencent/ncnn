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

#ifndef LAYER_CONVOLUTION_VULKAN_H
#define LAYER_CONVOLUTION_VULKAN_H

#include "convolution.h"

namespace ncnn {

class Convolution_vulkan : public Convolution
{
public:
    Convolution_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Convolution::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    ncnn::Layer* padding;

    Mat weight_data_packed;
    Mat weight_winograd23_data_packed;
    Mat weight_winograd43_data_packed;
    Mat bias_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    VkImageMat weight_data_gpu_image;
    VkImageMat bias_data_gpu_image;

    Pipeline* pipeline_convolution;
    Pipeline* pipeline_convolution_1x1s1d1;

    Pipeline* pipeline_convolution_gemm;

    // winograd23 and winograd43
    VkMat weight_data_gpu_tm_winograd23;
    VkImageMat weight_data_gpu_tm_winograd23_image;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_transform_input;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_gemm;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_transform_output;

    VkMat weight_data_gpu_tm_winograd43;
    VkImageMat weight_data_gpu_tm_winograd43_image;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_transform_input;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_gemm;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_transform_output;

    // convolution as fc
    ncnn::Layer* reshape_1x1xw;
    ncnn::Layer* reshape_w;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_VULKAN_H

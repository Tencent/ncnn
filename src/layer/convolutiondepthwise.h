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

#ifndef LAYER_CONVOLUTIONDEPTHWISE_H
#define LAYER_CONVOLUTIONDEPTHWISE_H

#include "layer.h"

namespace ncnn {

class ConvolutionDepthWise : public Layer
{
public:
    ConvolutionDepthWise();
    ~ConvolutionDepthWise();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

#if NCNN_VULKAN
    virtual int upload_model(VkTransfer& cmd);

    virtual int create_pipeline();
    virtual int destroy_pipeline();

    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int bias_term;

    int weight_data_size;
    int group;

    int int8_scale_term;

    // model
    Mat weight_data;
    Mat bias_data;

#if NCNN_VULKAN
    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    ncnn::Layer* padding;

    std::vector<ncnn::Layer*> convolution_group_ops;

    Pipeline* pipeline_convolutiondepthwise;

    Mat weight_data_pack4;
    Mat bias_data_pack4;
    VkMat weight_data_gpu_pack4;
    VkMat bias_data_gpu_pack4;
    Pipeline* pipeline_convolutiondepthwise_pack4;
#endif // NCNN_VULKAN

    Mat weight_data_int8_scales;
    Mat bottom_blob_int8_scales;

    bool use_int8_inference;

    std::vector<ncnn::Layer*> quantize_ops;
    std::vector<ncnn::Layer*> dequantize_ops;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_H

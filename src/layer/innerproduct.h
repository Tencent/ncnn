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

#ifndef LAYER_INNERPRODUCT_H
#define LAYER_INNERPRODUCT_H

#include "layer.h"

namespace ncnn {

class InnerProduct : public Layer
{
public:
    InnerProduct();
    ~InnerProduct();

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
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // model
    Mat weight_data;
    Mat bias_data;

#if NCNN_VULKAN
    ncnn::Layer* flatten;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_innerproduct;

    VkMat bias_data_gpu_pack4;

    // pack4
    VkMat weight_data_gpu_pack4;
    Pipeline* pipeline_innerproduct_pack4;

    // pack1to4
    VkMat weight_data_gpu_pack1to4;
    Pipeline* pipeline_innerproduct_pack1to4;

    // pack4to1
    VkMat weight_data_gpu_pack4to1;
    Pipeline* pipeline_innerproduct_pack4to1;
#endif // NCNN_VULKAN

    Mat weight_data_int8_scales;
    float bottom_blob_int8_scale;

    bool use_int8_inference;

    ncnn::Layer* quantize;
    std::vector<ncnn::Layer*> dequantize_ops;
};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_H

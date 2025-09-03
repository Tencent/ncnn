// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

public:
    Mat weight_data_packed;
    Mat weight_data_packed_groups;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    ncnn::Layer* padding;

    Pipeline* pipeline_convolutiondepthwise;
    Pipeline* pipeline_convolutiondepthwise_pack4;

    Pipeline* pipeline_convolutiondepthwise_group;
    Pipeline* pipeline_convolutiondepthwise_group_pack4;
    Pipeline* pipeline_convolutiondepthwise_group_pack1to4;
    Pipeline* pipeline_convolutiondepthwise_group_pack4to1;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_VULKAN_H

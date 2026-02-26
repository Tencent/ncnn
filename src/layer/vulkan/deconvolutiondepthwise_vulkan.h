// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTIONDEPTHWISE_VULKAN_H
#define LAYER_DECONVOLUTIONDEPTHWISE_VULKAN_H

#include "deconvolutiondepthwise.h"

namespace ncnn {

class DeconvolutionDepthWise_vulkan : public DeconvolutionDepthWise
{
public:
    DeconvolutionDepthWise_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using DeconvolutionDepthWise::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Mat weight_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    ncnn::Layer* crop;
    ncnn::Layer* output_crop;

    Pipeline* pipeline_deconvolutiondepthwise;
    Pipeline* pipeline_deconvolutiondepthwise_pack4;

    Pipeline* pipeline_deconvolutiondepthwise_group;
    Pipeline* pipeline_deconvolutiondepthwise_group_pack4;
    Pipeline* pipeline_deconvolutiondepthwise_group_pack1to4;
    Pipeline* pipeline_deconvolutiondepthwise_group_pack4to1;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTIONDEPTHWISE_VULKAN_H

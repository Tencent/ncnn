// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTIONDEPTHWISE1D_VULKAN_H
#define LAYER_CONVOLUTIONDEPTHWISE1D_VULKAN_H

#include "convolutiondepthwise1d.h"

namespace ncnn {

class ConvolutionDepthWise1D_vulkan : public ConvolutionDepthWise1D
{
public:
    ConvolutionDepthWise1D_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using ConvolutionDepthWise1D::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Mat weight_data_packed;
    Mat weight_data_packed_groups;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    ncnn::Layer* padding;

    Pipeline* pipeline_convolutiondepthwise1d;
    Pipeline* pipeline_convolutiondepthwise1d_pack4;
    Pipeline* pipeline_convolutiondepthwise1d_group;
    Pipeline* pipeline_convolutiondepthwise1d_group_pack4;
    Pipeline* pipeline_convolutiondepthwise1d_group_pack1to4;
    Pipeline* pipeline_convolutiondepthwise1d_group_pack4to1;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE1D_VULKAN_H

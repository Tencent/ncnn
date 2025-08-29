// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REQUANTIZE_VULKAN_H
#define LAYER_REQUANTIZE_VULKAN_H

#include "requantize.h"

namespace ncnn {

class Requantize_vulkan : virtual public Requantize
{
public:
    Requantize_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Requantize::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat scale_in_data_gpu;
    VkMat scale_out_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_requantize;
    Pipeline* pipeline_requantize_pack4;
};

} // namespace ncnn

#endif // LAYER_REQUANTIZE_VULKAN_H

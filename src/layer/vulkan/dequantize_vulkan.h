// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEQUANTIZE_VULKAN_H
#define LAYER_DEQUANTIZE_VULKAN_H

#include "dequantize.h"

namespace ncnn {

class Dequantize_vulkan : virtual public Dequantize
{
public:
    Dequantize_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Dequantize::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat scale_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_dequantize;
    Pipeline* pipeline_dequantize_pack4;
};

} // namespace ncnn

#endif // LAYER_DEQUANTIZE_VULKAN_H

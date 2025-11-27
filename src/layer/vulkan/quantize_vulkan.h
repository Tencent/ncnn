// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_QUANTIZE_VULKAN_H
#define LAYER_QUANTIZE_VULKAN_H

#include "quantize.h"

namespace ncnn {

class Quantize_vulkan : virtual public Quantize
{
public:
    Quantize_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Quantize::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat scale_data_gpu;

    Pipeline* pipeline_quantize;
    Pipeline* pipeline_quantize_pack4;
};

} // namespace ncnn

#endif // LAYER_QUANTIZE_VULKAN_H

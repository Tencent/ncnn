// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RELU_VULKAN_H
#define LAYER_RELU_VULKAN_H

#include "relu.h"

namespace ncnn {

class ReLU_vulkan : public ReLU
{
public:
    ReLU_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using ReLU::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_relu;
};

} // namespace ncnn

#endif // LAYER_RELU_VULKAN_H

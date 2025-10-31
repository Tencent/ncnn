// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELTWISE_VULKAN_H
#define LAYER_ELTWISE_VULKAN_H

#include "eltwise.h"

namespace ncnn {

class Eltwise_vulkan : public Eltwise
{
public:
    Eltwise_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Eltwise::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_eltwise[2];
};

} // namespace ncnn

#endif // LAYER_ELTWISE_VULKAN_H

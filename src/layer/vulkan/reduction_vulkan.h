// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REDUCTION_VULKAN_H
#define LAYER_REDUCTION_VULKAN_H

#include "reduction.h"

namespace ncnn {

class Reduction_vulkan : public Reduction
{
public:
    Reduction_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Reduction::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_reduction;
};

} // namespace ncnn

#endif // LAYER_REDUCTION_VULKAN_H

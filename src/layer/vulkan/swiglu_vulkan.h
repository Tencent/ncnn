// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWIGLU_VULKAN_H
#define LAYER_SWIGLU_VULKAN_H

#include "swiglu.h"

namespace ncnn {

class SwiGLU_vulkan : public SwiGLU
{
public:
    SwiGLU_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using SwiGLU::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_swiglu;
};

} // namespace ncnn

#endif // LAYER_SWIGLU_VULKAN_H

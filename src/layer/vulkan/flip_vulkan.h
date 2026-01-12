// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLIP_VULKAN_H
#define LAYER_FLIP_VULKAN_H

#include "flip.h"

namespace ncnn {

class Flip_vulkan : public Flip
{
public:
    Flip_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Flip::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_flip;
};

} // namespace ncnn

#endif // LAYER_FLIP_VULKAN_H

// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSWISH_VULKAN_H
#define LAYER_HARDSWISH_VULKAN_H

#include "hardswish.h"

namespace ncnn {

class HardSwish_vulkan : public HardSwish
{
public:
    HardSwish_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using HardSwish::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_hardswish;
};

} // namespace ncnn

#endif // LAYER_HARDSWISH_VULKAN_H

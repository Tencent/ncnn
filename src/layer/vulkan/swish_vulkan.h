// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWISH_VULKAN_H
#define LAYER_SWISH_VULKAN_H

#include "swish.h"

namespace ncnn {

class Swish_vulkan : public Swish
{
public:
    Swish_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Swish::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_swish;
};

} // namespace ncnn

#endif // LAYER_SWISH_VULKAN_H

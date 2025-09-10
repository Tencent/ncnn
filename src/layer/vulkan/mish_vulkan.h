// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MISH_VULKAN_H
#define LAYER_MISH_VULKAN_H

#include "mish.h"

namespace ncnn {

class Mish_vulkan : public Mish
{
public:
    Mish_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Mish::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_mish;
};

} // namespace ncnn

#endif // LAYER_MISH_VULKAN_H

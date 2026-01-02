// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTPLUS_VULKAN_H
#define LAYER_SOFTPLUS_VULKAN_H

#include "softplus.h"

namespace ncnn {

class Softplus_vulkan : public Softplus
{
public:
    Softplus_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Softplus::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_softplus;
};

} // namespace ncnn

#endif // LAYER_SOFTPLUS_VULKAN_H

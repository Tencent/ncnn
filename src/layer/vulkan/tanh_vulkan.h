// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TANH_VULKAN_H
#define LAYER_TANH_VULKAN_H

#include "tanh.h"

namespace ncnn {

class TanH_vulkan : public TanH
{
public:
    TanH_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using TanH::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_tanh;
};

} // namespace ncnn

#endif // LAYER_TANH_VULKAN_H

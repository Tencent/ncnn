// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSIGMOID_VULKAN_H
#define LAYER_HARDSIGMOID_VULKAN_H

#include "hardsigmoid.h"

namespace ncnn {

class HardSigmoid_vulkan : public HardSigmoid
{
public:
    HardSigmoid_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using HardSigmoid::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_hardsigmoid;
};

} // namespace ncnn

#endif // LAYER_HARDSIGMOID_VULKAN_H

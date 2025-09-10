// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEEPCOPY_VULKAN_H
#define LAYER_DEEPCOPY_VULKAN_H

#include "deepcopy.h"

namespace ncnn {

class DeepCopy_vulkan : public DeepCopy
{
public:
    DeepCopy_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using DeepCopy::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_deepcopy;
    Pipeline* pipeline_deepcopy_pack4;
};

} // namespace ncnn

#endif // LAYER_DEEPCOPY_VULKAN_H

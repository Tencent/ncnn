// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLATTEN_VULKAN_H
#define LAYER_FLATTEN_VULKAN_H

#include "flatten.h"

namespace ncnn {

class Flatten_vulkan : public Flatten
{
public:
    Flatten_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Flatten::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_flatten;
    Pipeline* pipeline_flatten_pack4;
    Pipeline* pipeline_flatten_pack1to4;
};

} // namespace ncnn

#endif // LAYER_FLATTEN_VULKAN_H
